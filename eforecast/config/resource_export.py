# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Export the contract of the Runs page's resource report; then npm run contracts:generate in webui."""
import argparse
import json
from pathlib import Path
from .resource_contract import ResourceContract
from .schema import JobsConfig

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'schema/resource_contract.schema.json'


def render():
    schema = ResourceContract.model_json_schema()
    schema['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
    schema['$id'] = 'https://github.com/joesider9/ProjectPipelineTest/schema/resource_contract.schema.json'
    schema['x-case-defaults'] = {k: JobsConfig.model_fields[k].default
                              for k in schema['$defs']['ResourceControls']['properties']}
    return json.dumps(schema, indent=2) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = render()
    if args.check:
        if not PATH.is_file() or PATH.read_text() != content:
            raise SystemExit('resource schema stale: python -m eforecast.config.resource_export')
        print('resource schema current')
    else:
        PATH.write_text(content)
        print('wrote ' + str(PATH))


if __name__ == '__main__':
    main()
