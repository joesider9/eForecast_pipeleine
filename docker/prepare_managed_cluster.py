# eForecast project author: George Sideratos and Claude
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Prepare the isolated qualification target after disposable services are started.

Connection values are written only to protected scratch files and the Kubernetes
Secret. This script never changes the live lab target or provisions cloud resources.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scratch',required=True,type=Path)
    p.add_argument('--container',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--port',type=int,default=56443)
    p.add_argument('--namespace',default='eforecast-cpu-pilot')
    p.add_argument('--node',default='eforecast-integration-cpu')
    p.add_argument('--gpu',type=int,choices=(0,1),default=0)
    p.add_argument('--cpus',type=int,default=1)
    p.add_argument('--ram-mb',type=int,default=3072)
    a=p.parse_args();r=a.scratch.resolve();os.umask(0o077)
    assert 'integration' in str(r) and a.container.startswith('eforecast-kube-integration')
    def k(*argv,input=None):
        return subprocess.check_output(['docker','exec','-i',a.container,'kubectl',*argv],input=input,text=True).strip()
    for source in ('00-namespace.yaml','10-rbac.yaml'):
        documents=list(yaml.safe_load_all(Path('deploy/kubernetes',source).read_text()))
        for d in documents:
            if d['kind']=='Namespace': d['metadata']['name']=a.namespace
            elif 'namespace' in d['metadata']: d['metadata']['namespace']=a.namespace
            for subject in d.get('subjects',[]):
                if 'namespace' in subject:subject['namespace']=a.namespace
        k('apply','-f','-',input=yaml.safe_dump_all(documents))
    (r/'token').write_text(k('-n',a.namespace,'create','token','eforecast-executor','--duration=4h'))
    (r/'ca.crt').write_bytes(subprocess.check_output(['docker','exec',a.container,'cat','/var/lib/rancher/k3s/server/tls/server-ca.crt']))
    ips={t:json.loads(subprocess.check_output(['docker','inspect',a.container+'-'+t]))[0]['NetworkSettings']['Networks'][a.container]['IPAddress'] for t in ('pg','rabbit')}
    dsn=f"postgresql://postgres@{ips['pg']}/postgres";pw=json.loads((r/'rabbit-private.json').read_text())['password']
    allocator_namespace=a.namespace+'-compute'
    k('apply','-f','-',input=json.dumps({'apiVersion':'v1','kind':'Secret','metadata':{'name':'managed-connections','namespace':a.namespace},'type':'Opaque','stringData':{'EFORECAST_RUN_STORE_DSN':dsn,'EFORECAST_ALLOCATOR_DSN':dsn,'EFORECAST_ALLOCATOR_NAMESPACE':allocator_namespace}}))
    env={'EFORECAST_WORKSPACE':str(r/'shared/workspace'),'EFORECAST_RUN_STORE':'postgres','EFORECAST_RUN_STORE_DSN':dsn,'EFORECAST_RUN_STORE_NAMESPACE':a.namespace,'EFORECAST_ALLOCATOR_BACKEND':'postgres','EFORECAST_ALLOCATOR_DSN':dsn,'EFORECAST_ALLOCATOR_NAMESPACE':allocator_namespace,'EFORECAST_BROKER_URL':f"amqp://pilot:{pw}@{ips['rabbit']}:5672/%2F",'EFORECAST_KUBE_SERVER':f'https://127.0.0.1:{a.port}','EFORECAST_KUBE_NAMESPACE':a.namespace,'EFORECAST_KUBE_CA':str(r/'ca.crt'),'EFORECAST_KUBE_TOKEN_FILE':str(r/'token'),'EFORECAST_KUBE_RUNS_FILE':str(r/'target.json'),'EFORECAST_RUN_EXECUTOR_DIR':str(r/'shared/state/host')}
    (r/'host-env.json').write_text(json.dumps(env,indent=2))
    target=json.loads(Path('deploy/kubernetes/managed-run.example.json').read_text());target.update(image=a.image,server=env['EFORECAST_KUBE_SERVER'],namespace=a.namespace,shared_root=str(r/'shared'),workspace=env['EFORECAST_WORKSPACE'],state_root=str(r/'shared/state'),resources={'cpu_threads':a.cpus,'ram_mb':a.ram_mb,'gpu_slots':a.gpu},node_selector={'kubernetes.io/hostname':a.node},active_deadline=600,max_jobs=1 if a.gpu==0 else 2,volume={'hostPath':{'path':str(r/'shared'),'type':'Directory'}})
    (r/'target.json').write_text(json.dumps(target,indent=2));print('Private connections, pinned target and namespace permissions prepared')


if __name__=='__main__':main()
