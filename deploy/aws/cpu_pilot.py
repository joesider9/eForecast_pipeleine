# eForecast project author: George Sideratos and Claude
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""One explicitly bounded EC2 qualification host; never a production deployer.

The caller supplies a validated CPU-only envelope, a fresh public SSH key, a /32
operator address and an inventory path outside the repository. Cloud credentials
stay on the caller. `destroy` only acts on exact IDs bearing this pilot's tag.
"""
from datetime import datetime, timedelta, timezone
import argparse
import ipaddress
import json
import os
from pathlib import Path
import time
import uuid


def save(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def userdata(deadline):
    import yaml
    expires = (deadline - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S UTC')
    return '#cloud-config\n' + yaml.safe_dump({
        'write_files': [
            {'path': '/etc/systemd/system/eforecast-pilot-expire.service', 'content':
             '[Unit]\nDescription=Terminate bounded eforecast pilot\n[Service]\nType=oneshot\nExecStart=/usr/bin/systemctl poweroff\n'},
            {'path': '/etc/systemd/system/eforecast-pilot-expire.timer', 'content':
             '[Unit]\nDescription=Fixed pilot deadline, including reboot\n[Timer]\n'
             f'OnCalendar={expires}\nPersistent=true\nAccuracySec=1s\n'
             '[Install]\nWantedBy=timers.target\n'}],
        'runcmd': [['systemctl', 'daemon-reload'],
                   ['systemctl', 'enable', '--now', 'eforecast-pilot-expire.timer'],
                   ['sh', '-c', 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io python3-venv && systemctl enable --now docker']]})


def main():
    from eforecast.venues.aws_pilot import validate
    import boto3
    from botocore.config import Config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'inspect', 'destroy'))
    parser.add_argument('--envelope', required=True, type=Path)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--public-key', type=Path)
    parser.add_argument('--operator-cidr')
    parser.add_argument('--subnet')
    parser.add_argument('--ami')
    args = parser.parse_args()
    document = json.loads(args.envelope.read_text())
    gpu_only = document.get('gpu') == {'instance_type': 'g5.xlarge', 'nodes': 1} and document.get('cpu') == {'instance_type': None, 'nodes': 0}
    cpu_only = document.get('gpu') == {'instance_type': None, 'nodes': 0} and document.get('cpu', {}).get('nodes') == 1 and document['cpu'].get('instance_type') in ('m6i.xlarge', 'm7i-flex.large')
    if args.action == 'launch':
        validate(document)
        assert gpu_only or cpu_only, 'one supported CPU or GPU instance only'
        assert document['spending_limit_usd'] <= 50 and document['max_duration_seconds'] <= 18000
        assert args.public_key and args.subnet and args.ami
        network = ipaddress.ip_network(args.operator_cidr)
        assert network.version == 4 and network.prefixlen == 32
        assert not args.inventory.exists(), 'Inspect/clean the existing inventory; never overwrite it'
    session = boto3.Session(profile_name=document['profile'], region_name=document['region'])
    cfg = Config(connect_timeout=5, read_timeout=20, retries={'max_attempts': 1})
    def client(service):
        return session.client(service, config=cfg)
    assert client('sts').get_caller_identity()['Account'] == document['account_id']
    if args.action == 'launch' and gpu_only:
        # The account may have a G/VT quota yet still be on the Free plan,
        # which refuses every non-Free-Tier EC2 type. Check before creating
        # the security group, SSH key, or shutdown role.
        plan = session.client('freetier', region_name='us-east-1', config=cfg).get_account_plan_state()
        if plan.get('accountPlanType') != 'PAID':
            raise RuntimeError('GPU launch requires an AWS Paid account plan; current plan is '
                               + str(plan.get('accountPlanType')))
    ec2, iam, scheduler = client('ec2'), client('iam'), client('scheduler')
    if args.action == 'launch':
        started = datetime.now(timezone.utc)
        deadline = datetime.fromisoformat(document['deadline_utc'].replace('Z', '+00:00'))
        assert timedelta(minutes=30) < deadline-started <= timedelta(hours=5)
        name = ('eforecast-gpu-' if gpu_only else 'eforecast-cpu-') + uuid.uuid4().hex[:12]
        inventory = {'name': name, 'account': document['account_id'], 'region': document['region'],
            'started_utc': started.isoformat(), 'deadline_utc': deadline.isoformat(),
            'status': 'creating', 'resources': {}, 'events': [], 'shared': {'subnet': args.subnet},
            'envelope': str(args.envelope.resolve())}
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        save(args.inventory, inventory)
    else:
        inventory = json.loads(args.inventory.read_text()); name = inventory['name']
        assert inventory['account'] == document['account_id'] and inventory['region'] == document['region']
    owned = inventory['resources']
    def remember(kind, identity):
        owned[kind] = identity
        inventory['events'].append({'at': datetime.now(timezone.utc).isoformat(), 'created': kind, 'id': identity})
        save(args.inventory, inventory)
    tags = [{'Key': 'eforecast-pilot', 'Value': name}, {'Key': 'Name', 'Value': name}]
    def exact_tag(values):
        return any(t['Key'] == 'eforecast-pilot' and t['Value'] == name for t in values)
    if args.action == 'launch':
        subnet = ec2.describe_subnets(SubnetIds=[args.subnet])['Subnets'][0]
        inventory['shared']['vpc'] = subnet['VpcId'];save(args.inventory, inventory)
        remember('security_group', ec2.create_security_group(GroupName=name, Description='Disposable pilot SSH from operator only',
            VpcId=subnet['VpcId'], TagSpecifications=[{'ResourceType':'security-group','Tags':tags}])['GroupId'])
        ec2.authorize_security_group_ingress(GroupId=owned['security_group'], IpPermissions=[{
            'IpProtocol':'tcp','FromPort':22,'ToPort':22,'IpRanges':[{'CidrIp':str(network),'Description':'Pilot operator'}]}])
        ec2.import_key_pair(KeyName=name, PublicKeyMaterial=args.public_key.read_bytes(),
            TagSpecifications=[{'ResourceType':'key-pair','Tags':tags}]);remember('key_pair', name)
        trust={'Version':'2012-10-17','Statement':[{'Effect':'Allow','Principal':{'Service':'scheduler.amazonaws.com'},
            'Action':'sts:AssumeRole','Condition':{'StringEquals':{'aws:SourceAccount':document['account_id']},
            'ArnEquals':{'aws:SourceArn':f"arn:aws:scheduler:{document['region']}:{document['account_id']}:schedule-group/default"}}}]}
        role = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),Tags=tags)['Role']
        remember('role', name)
        policy={'Version':'2012-10-17','Statement':[{'Effect':'Allow','Action':'ec2:TerminateInstances',
            'Resource':f"arn:aws:ec2:{document['region']}:{document['account_id']}:instance/*",
            'Condition':{'StringEquals':{'ec2:ResourceTag/eforecast-pilot':name}}}]}
        iam.put_role_policy(RoleName=name, PolicyName='terminate-owned-pilot',PolicyDocument=json.dumps(policy))
        launch=dict(ImageId=args.ami,InstanceType=document['gpu']['instance_type'] if gpu_only else document['cpu']['instance_type'],MinCount=1,MaxCount=1,ClientToken=name,
            KeyName=name,NetworkInterfaces=[{'DeviceIndex':0,'SubnetId':args.subnet,'Groups':[owned['security_group']],
                'AssociatePublicIpAddress':True,'DeleteOnTermination':True}],
            BlockDeviceMappings=[{'DeviceName':'/dev/sda1','Ebs':{'VolumeSize':80,'VolumeType':'gp3','Encrypted':True,'DeleteOnTermination':True}}],
            # cloud-init needs IMDS for its public SSH key and bootstrap timer.
            # No instance profile is attached; require tokens and one network hop.
            MetadataOptions={'HttpEndpoint':'enabled','HttpTokens':'required','HttpPutResponseHopLimit':1},InstanceInitiatedShutdownBehavior='terminate',
            UserData=userdata(deadline),TagSpecifications=[{'ResourceType':t,'Tags':tags} for t in ('instance','volume')])
        save(args.inventory.with_name('launch-request.json'),launch)
        instances=ec2.run_instances(**launch)['Instances'];assert len(instances)==1
        remember('instance', instances[0]['InstanceId'])
        # If the independent shutdown cannot be installed, terminate immediately.
        try:
            due=(deadline-timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%S')
            request=dict(Name=name,ScheduleExpression=f'at({due})',ScheduleExpressionTimezone='UTC',
                FlexibleTimeWindow={'Mode':'OFF'},ActionAfterCompletion='DELETE',
                Target={'Arn':'arn:aws:scheduler:::aws-sdk:ec2:terminateInstances','RoleArn':role['Arn'],
                    'Input':json.dumps({'InstanceIds':[owned['instance']]}),
                    'RetryPolicy':{'MaximumEventAgeInSeconds':60,'MaximumRetryAttempts':0}})
            save(args.inventory.with_name('shutdown-schedule-request.json'),request)
            for attempt in range(8):
                try:
                    scheduler.create_schedule(**request);break
                except scheduler.exceptions.ValidationException:
                    if attempt==7: raise
                    time.sleep(3)
            remember('schedule',name)
            scheduler.get_schedule(Name=name)
        except BaseException:
            ec2.terminate_instances(InstanceIds=[owned['instance']]);raise
        inventory['status']='launched';save(args.inventory,inventory)
        print(json.dumps({'instance':owned['instance'],'deadline_utc':inventory['deadline_utc'],'inventory':str(args.inventory)}))
    elif args.action == 'inspect':
        items=ec2.describe_instances(InstanceIds=[owned['instance']])['Reservations'][0]['Instances'] if owned.get('instance') else []
        print(json.dumps({'inventory':inventory,'instances':[{k:i.get(k) for k in ('InstanceId','State','PublicIpAddress','BlockDeviceMappings')} for i in items]},default=str,indent=2))
    else:
        if owned.get('instance'):
            instances=ec2.describe_instances(InstanceIds=[owned['instance']])['Reservations'][0]['Instances']
            assert len(instances)==1 and exact_tag(instances[0]['Tags'])
            ec2.terminate_instances(InstanceIds=[owned['instance']])
            ec2.get_waiter('instance_terminated').wait(InstanceIds=[owned['instance']],WaiterConfig={'Delay':5,'MaxAttempts':60})
        if owned.get('schedule'):
            try: scheduler.delete_schedule(Name=owned['schedule'])
            except scheduler.exceptions.ResourceNotFoundException: pass
        if owned.get('role'):
            role=iam.get_role(RoleName=owned['role'])['Role'];assert exact_tag(role['Tags'])
            iam.delete_role_policy(RoleName=owned['role'],PolicyName='terminate-owned-pilot');iam.delete_role(RoleName=owned['role'])
        if owned.get('key_pair'):
            keys=ec2.describe_key_pairs(KeyNames=[owned['key_pair']])['KeyPairs'];assert len(keys)==1 and exact_tag(keys[0]['Tags'])
            ec2.delete_key_pair(KeyName=owned['key_pair'])
        if owned.get('security_group'):
            groups=ec2.describe_security_groups(GroupIds=[owned['security_group']])['SecurityGroups'];assert len(groups)==1 and exact_tag(groups[0]['Tags'])
            ec2.delete_security_group(GroupId=owned['security_group'])
        volumes=ec2.describe_volumes(Filters=[{'Name':'tag:eforecast-pilot','Values':[name]}])['Volumes']
        assert not volumes, 'Owned volumes remain: retain inventory and investigate'
        inventory['status']='removed';inventory['removed_utc']=datetime.now(timezone.utc).isoformat();save(args.inventory,inventory)
        print(json.dumps({'status':'removed','pilot':name,'remaining_volumes':len(volumes)}))


if __name__ == '__main__':
    main()
