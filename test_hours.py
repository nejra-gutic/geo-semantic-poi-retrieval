from datetime import datetime
from src.retrieval.hours import is_open_now, parse_opening_hours

test_cases = [
    'Mo-Fr 09:00-17:00',
    'Mo-Sa 07:30-20:00; Su 07:30-18:30',
    '24/7',
    'Mo-Fr 09:00-17:00; Sa 09:00-13:00; Su closed',
    'Mo-Fr 08:00-14:00,17:30-22:00; Sa 09:00-14:30,17:30-22:00',
    'Tu-Th,Su 17:00-22:00; Fr,Sa 17:00-23:00',
]

for h in test_cases:
    parsed = parse_opening_hours(h)
    open_now = is_open_now(h)
    print(f"{h!r:60} parsed={parsed is not None}  open_now={open_now}")
