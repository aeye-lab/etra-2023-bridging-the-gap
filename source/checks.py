from typing import Dict


def check_format_spec(format_spec: Dict[str, int]) -> None:
    '''
    check format specification for numpy channels/columns.

    - check if all values are unique.
    - check that lowest value is zero
    - check for no gaps
    - check that highest value is len(format_spec) - 1
    '''

    # we can do this all by iterating over range
    for idx in range(len(format_spec)):
        if idx not in format_spec.values():
            raise ValueError(f'{idx} not in format_spec')
