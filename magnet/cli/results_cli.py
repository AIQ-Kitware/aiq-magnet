"""Commands for persisted results; none execute claims or request computation."""

import json
import os
from pathlib import Path

import kwconf
import safer

from magnet.records import RecordFiles
from magnet.results import (
    discover_result_bundles,
    export_result_bundle,
    load_result_bundle,
    render_result,
    verify_result_bundle,
)


class ShowResultCLI(kwconf.Config):
    """Present an existing native result, including older dashboard bundles."""

    path = kwconf.Value(None, required=True, position=1)
    format = kwconf.Value('markdown', choices=['markdown', 'text', 'json'])
    output = kwconf.Value(
        None, help='Write the presentation here instead of stdout'
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True)
        bundle = load_result_bundle(args.path)
        text = render_result(bundle, format=args.format)
        if args.output:
            target = Path(args.output).resolve()
            root = bundle.root.resolve()
            if (
                target.is_relative_to(root)
                and target.relative_to(root).as_posix()
                not in RecordFiles.REPORTS
            ):
                raise ValueError(
                    'presentation must not replace native result files'
                )
            with safer.open(
                target, 'w', encoding='utf-8', temp_file=os.name != 'nt'
            ) as file:
                file.write(text)
        else:
            print(text, end='')


class ExportResultCLI(kwconf.Config):
    """Export a native result with optional selected supporting artifacts."""

    source = kwconf.Value(None, required=True, position=1)
    destination = kwconf.Value(None, required=True, position=2)
    supporting = kwconf.Value('none', choices=['none', 'terminal', 'requested'])
    nodes = kwconf.Value(
        '', help='Comma-separated requested node names; empty selects all'
    )
    include_logs = kwconf.Flag(False)
    attachments = kwconf.Value(
        None,
        help='JSON/YAML mapping (or mapping file) of attachment names to files',
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True)
        attachments = None
        if args.attachments:
            import kwutil

            attachments = kwutil.Yaml.coerce(args.attachments)
            if not isinstance(attachments, dict):
                raise ValueError('attachments must map relative names to files')
        result = export_result_bundle(
            args.source,
            args.destination,
            supporting=args.supporting,
            nodes=tuple(x.strip() for x in args.nodes.split(',') if x.strip()),
            include_logs=args.include_logs,
            attachments=attachments,
        )
        print(result)


class VerifyResultCLI(kwconf.Config):
    """Verify an exported bundle's bytes and claim references."""

    path = kwconf.Value(None, required=True, position=1)

    @classmethod
    def main(cls, argv=None, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True)
        result = verify_result_bundle(args.path)
        print(json.dumps(result, indent=2))
        return 0 if result['valid'] else 1


class ListResultsCLI(kwconf.Config):
    """Find saved evaluation records without loading pipeline code or caches."""

    path = kwconf.Value(None, required=True, position=1)
    format = kwconf.Value('text', choices=['text', 'json'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True)
        records = discover_result_bundles(args.path)
        if args.format == 'json':
            print(json.dumps(records, indent=2))
        else:
            for item in records:
                print(
                    f'{item["result"]:<15} claims={item["available_claims"]}/{item["claims"]} '
                    f'execution={item["execution_state"]}  {item["path"]}'
                )


class ResultsCLI(kwconf.ModalCLI):
    """Read, present, export and check existing serialized evaluation results."""

    show = ShowResultCLI
    export = ExportResultCLI
    verify = VerifyResultCLI
    list = ListResultsCLI
