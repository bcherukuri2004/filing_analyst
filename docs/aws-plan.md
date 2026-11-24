# Running filing_analyst on AWS: the plan

Status: **a plan only. Nothing has been created on AWS.** Every command below
waits for your approval of the exact cost and the exact commands.

## The setup in one paragraph

One small Linux server (an EC2 instance, type `t4g.small`) runs the same
`docker compose` stack that already works on your Mac: Postgres with pgvector,
the one time ingest step, and the API. The server has **no open ports at all**.
You reach the API from your Mac through an encrypted tunnel made by AWS Systems
Manager (SSM), so nobody else on the internet can even see it. A spending alert
emails you if the month goes over $20. One script deletes everything.

## Why this setup

| Option | Rough monthly cost (`us-east-1`, running all month) | Why or why not |
|---|---|---|
| **One EC2 `t4g.small`, database in a container** (picked) | about **$17.50** | Cheapest real server. Same compose file as on your Mac, so nothing new can break. ARM (Graviton) like your Mac, so the image needs no changes |
| One EC2 `t4g.medium` | about $30 | Twice the memory (4 GB). The fallback if 2 GB turns out too tight |
| ECS Fargate (2 tasks) + RDS Postgres + a load balancer | $70 or more | The "proper" managed setup, but the load balancer alone costs more than the whole picked setup |

Memory check for `t4g.small` (2 GB): building the index peaked at about
1.2 GB in Docker on your Mac, and the API is not running at that time (it waits
for the ingest step). The plan adds a 2 GB swap file as a safety net. If the
ingest step still runs out of memory, the fix is switching to `t4g.medium`.

## Cost, line by line

Prices are for `us-east-1` (North Virginia), Linux, on demand, from third party
price trackers ([ec2calc](https://ec2calc.cloud/instances/t4g.small),
[economize](https://www.economize.cloud/resources/aws/pricing/ec2/t4g.medium/),
[Vantage on IPv4](https://www.vantage.sh/blog/aws-public-ipv4-cost-and-how-to-see)).
**They are not yet confirmed against AWS's own pages.** Step 0 below confirms
them with AWS's official price list before anything is created; if they
differ, the numbers here get updated and you approve again.

| Item | Price | A full month (730 hours) | One day, then deleted |
|---|---|---|---|
| EC2 `t4g.small` | $0.0168 an hour | $12.26 | $0.40 |
| Public IPv4 address (needed so the server can reach GitHub, Docker Hub and the SEC; no inbound ports are open) | $0.005 an hour | $3.65 | $0.12 |
| 20 GB disk (EBS gp3) | $0.08 per GB a month (one source says $0.088) | $1.60 (or $1.76) | about $0.06 |
| CPU above the burst baseline (only while building the index, once) | $0.04 per extra vCPU hour | under $0.05 | under $0.05 |
| Systems Manager tunnel, data sent to you (first 100 GB a month), spending alert | free | $0 | $0 |
| **Total** | | **about $17.50 to $17.70** | **about $0.65** |

Credits: AWS says new accounts get $100 in credits (up to $200 with
activities), per the [AWS billing docs](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier.html).
An existing account does not get them. Either way, the alert at $20 is set
before the server exists.

## What you need first (on your Mac)

* An AWS account you can sign in to. **You** sign in; I never type, see or
  store a password or an access key.
* Two tools from Homebrew (about 100 MB together): the AWS command line, and
  the Session Manager plugin that makes the tunnel:

  ```
  brew install awscli
  brew install --cask session-manager-plugin
  ```

* The reviewed branches pushed to GitHub, because the server downloads the code
  from `github.com/bcherukuri2004/filing_analyst`. That push is also when the
  new CI smoke test first runs on GitHub.

## The exact steps

All commands use one region and one tag, so every piece can be found and
deleted later:

```
export AWS_REGION=us-east-1
export TAG="Key=project,Value=filing-analyst"
```

### 0. Sign in (you), then confirm the prices (free)

```
aws configure sso          # you do this; it opens a browser sign in. No keys in files.
aws sts get-caller-identity --query Account --output text   # shows which account

# Official AWS price for t4g.small in us-east-1 (the Price List API is free):
aws pricing get-products --region us-east-1 --service-code AmazonEC2 --filters \
  Type=TERM_MATCH,Field=instanceType,Value=t4g.small \
  Type=TERM_MATCH,Field=location,Value="US East (N. Virginia)" \
  Type=TERM_MATCH,Field=operatingSystem,Value=Linux \
  Type=TERM_MATCH,Field=tenancy,Value=Shared \
  Type=TERM_MATCH,Field=preInstalledSw,Value=NA \
  Type=TERM_MATCH,Field=capacitystatus,Value=Used \
  --query 'PriceList[0]' --output text | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); \
  print([p['pricePerUnit'] for t in d['terms']['OnDemand'].values() for p in t['priceDimensions'].values()])"
```

### 1. A spending alert at $20 (before anything that costs money)

```
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws budgets create-budget --account-id "$ACCOUNT" \
  --budget '{"BudgetName":"filing-analyst","BudgetLimit":{"Amount":"20","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80,"ThresholdType":"PERCENTAGE"},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"<your email>"}]}]'
```

### 2. Permission for the server to accept the tunnel (no SSH, no keys)

```
aws iam create-role --role-name filing-analyst-ssm --tags "$TAG" \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name filing-analyst-ssm \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam create-instance-profile --instance-profile-name filing-analyst-ssm --tags "$TAG"
aws iam add-role-to-instance-profile --instance-profile-name filing-analyst-ssm --role-name filing-analyst-ssm
```

### 3. A firewall with nothing open

```
SG=$(aws ec2 create-security-group --group-name filing-analyst \
  --description "filing-analyst: no inbound ports" \
  --tag-specifications "ResourceType=security-group,Tags=[{$TAG}]" --query GroupId --output text)
# No "authorize-security-group-ingress" at all: nothing can connect in.
```

### 4. The server

The setup script (`deploy/aws-user-data.sh`, run once by the server when it
first starts) installs Docker, adds a 2 GB swap file, downloads the code and
the filings, makes a random database password and API key **on the server**
(they never leave it), and starts `docker compose`.

```
# Fill in the two placeholders on your Mac (the file in git keeps them blank).
export SEC_USER_AGENT="Your Name you@example.com"
sed -e "s/__BRANCH__/main/" -e "s/__SEC_USER_AGENT__/$SEC_USER_AGENT/" \
  deploy/aws-user-data.sh > /tmp/filing-analyst-user-data.sh

AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
  --query Parameter.Value --output text)
INSTANCE=$(aws ec2 run-instances --image-id "$AMI" --instance-type t4g.small \
  --iam-instance-profile Name=filing-analyst-ssm --security-group-ids "$SG" \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=20,VolumeType=gp3,DeleteOnTermination=true}' \
  --metadata-options HttpTokens=required \
  --user-data file:///tmp/filing-analyst-user-data.sh \
  --tag-specifications "ResourceType=instance,Tags=[{$TAG}]" "ResourceType=volume,Tags=[{$TAG}]" \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --instance-ids "$INSTANCE"
```

### 5. Use it from your Mac

```
# The tunnel: http://127.0.0.1:8000 on your Mac reaches the API on the server.
aws ssm start-session --target "$INSTANCE" --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8000"],"localPortNumber":["8000"]}'

# In another terminal:
curl -s 127.0.0.1:8000/health
```

To get the API key for `/search`, open a shell on the server through the same
tunnel service (no SSH) and read it there: `aws ssm start-session --target
"$INSTANCE"`, then `sudo grep API_KEY /opt/filing_analyst/.env`.

What would be measured and reported, as before: time from `run-instances` to
the first healthy `/health`, the ingest time on this server, `/search` answers,
memory in use, and the cost shown in AWS Cost Explorer the next day.

## Shutting everything down

`deploy/aws-down.sh` runs these, in this order (each waits for the one before):

```
aws ec2 terminate-instances --instance-ids "$INSTANCE"
aws ec2 wait instance-terminated --instance-ids "$INSTANCE"     # its disk is deleted with it
aws ec2 delete-security-group --group-id "$SG"
aws iam remove-role-from-instance-profile --instance-profile-name filing-analyst-ssm --role-name filing-analyst-ssm
aws iam delete-instance-profile --instance-profile-name filing-analyst-ssm
aws iam detach-role-policy --role-name filing-analyst-ssm --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam delete-role --role-name filing-analyst-ssm
aws budgets delete-budget --account-id "$ACCOUNT" --budget-name filing-analyst   # optional: keep the alert

# Proof nothing is left: this must print nothing.
aws resourcegroupstaggingapi get-resources --tag-filters Key=project,Values=filing-analyst \
  --query 'ResourceTagMappingList[].ResourceARN' --output text
aws ec2 describe-instances --filters Name=tag:project,Values=filing-analyst \
  Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text
```

AWS's tag search can keep listing a just deleted server for a while (up to
about an hour); the second check, which only counts servers that still exist,
is the one that matters.

Stopping the server instead of deleting it is cheaper but not free: the disk
($1.60 a month) still costs while stopped. Deleting is the only way to $0.

## Not decided yet (your call)

1. **Can I push the reviewed branches to GitHub?** The server downloads the code
   from there.
2. **Tunnel only, or a public address?** Picked: tunnel only (nothing open).
   A public address would need a small change to the compose file (it only
   listens on 127.0.0.1 today) and a firewall rule for your IP only.
3. **Region:** `us-east-1` is picked as one of the cheapest.
4. **How long it runs:** one day then deleted (about $0.65), or kept running
   (about $17.50 a month).
