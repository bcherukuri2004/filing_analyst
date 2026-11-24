#!/usr/bin/env bash
# Delete everything docs/aws-plan.md creates, then prove nothing is left.
# NOT RUN YET: waiting for approval, like the rest of the plan.
#
#   deploy/aws-down.sh            delete everything but keep the $20 alert
#   deploy/aws-down.sh --budget   also delete the $20 alert
set -uo pipefail   # no -e: keep going, so one missing piece does not stop the cleanup
export AWS_REGION="${AWS_REGION:-us-east-1}"

for id in $(aws ec2 describe-instances --filters Name=tag:project,Values=filing-analyst \
    Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[].Instances[].InstanceId' --output text); do
  echo "terminating $id (its disk is deleted with it)"
  aws ec2 terminate-instances --instance-ids "$id" >/dev/null
  aws ec2 wait instance-terminated --instance-ids "$id"
done

SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=filing-analyst \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
[ -n "$SG" ] && [ "$SG" != "None" ] && aws ec2 delete-security-group --group-id "$SG" && echo "deleted firewall $SG"

aws iam remove-role-from-instance-profile --instance-profile-name filing-analyst-ssm --role-name filing-analyst-ssm 2>/dev/null
aws iam delete-instance-profile --instance-profile-name filing-analyst-ssm 2>/dev/null && echo "deleted instance profile"
aws iam detach-role-policy --role-name filing-analyst-ssm \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null
aws iam delete-role --role-name filing-analyst-ssm 2>/dev/null && echo "deleted role"

if [ "${1:-}" = "--budget" ]; then
  ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
  aws budgets delete-budget --account-id "$ACCOUNT" --budget-name filing-analyst && echo "deleted the \$20 alert"
fi

echo "== anything still tagged project=filing-analyst (should be nothing):"
left=$(aws resourcegroupstaggingapi get-resources --tag-filters Key=project,Values=filing-analyst \
  --query 'ResourceTagMappingList[].ResourceARN' --output text)
running=$(aws ec2 describe-instances --filters Name=tag:project,Values=filing-analyst \
  Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text)
[ -n "$left" ] && echo "tag search still lists: $left (it can lag up to about an hour after a delete)"
if [ -z "$running" ]; then echo "no filing-analyst server exists"; else echo "SERVER STILL THERE: $running"; exit 1; fi
