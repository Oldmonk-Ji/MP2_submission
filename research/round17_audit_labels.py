"""New-parent text-only audit annotations, before model prediction.

Ambiguous price/total and weekend bases remain masked. Heads do not resolve
Alice/Bob acorn ownership, croissant varieties or allowance time stages.
These limitations prevent interpreting the diagnostic as full relation gold.
"""
from round17_labels import OUT,write_labels
AUDIT={
'909:-1':'allowance:O allowance:P allowance:O allowance:MP',
'909:0':'allowance:O allowance:P allowance:O allowance:MP allowance:P allowance:P allowance:O allowance:O',
'909:1':'allowance:O allowance:P allowance:O allowance:P allowance:O allowance:MP allowance:O allowance:O',
'2259:-1':'acorn:C acorn:? acorn:T',
'2259:0':'acorn:C acorn:? acorn:T acorn:T acorn:P acorn:MP acorn:P acorn:P',
'2259:3':'acorn:C acorn:? acorn:T acorn:P acorn:MP acorn:P acorn:P acorn:P',
'1532:-1':'weekday:P weekend:?',
'1532:0':'weekday:C weekend:C',
'1532:1':'weekday:P weekend:? ! weekday:P weekday:T weekday:T weekend:T restaurant:T',
'1023:-1':'?:T broccoli:T carrot:MA spinach:O spinach:MA',
'1023:0':'broccoli:T carrot:MA carrot:T spinach:O spinach:T spinach:T ?:T ?:T',
'1023:1':'broccoli:T carrot:MA spinach:O carrot:T ! broccoli:T spinach:T ?:T',
'4371:-1':'cherry:P genevieve:O genevieve:T',
'4371:0':'cherry:P genevieve:T genevieve:O genevieve:T genevieve:O cherry:T cherry:P',
'4371:3':'cherry:P genevieve:O ! ! genevieve:O !',
'1819:-1':'system:T zone:C vent:C',
'1819:1':'system:T zone:C vent:C system:T zone:C vent:C vent:C vent:P',
'1819:3':'system:T zone:C vent:C system:T vent:C vent:P',
'2694:-1':'croissant:C croissant:C croissant:P croissant:P focaccia:P latte:C latte:P',
'2694:1':'croissant:C croissant:P focaccia:C focaccia:P',
'2694:2':'croissant:C croissant:P focaccia:C focaccia:P',
'879:-1':'josie:T tape:C tape:P headphone:P',
'879:0':'tape:C tape:P tape:P tape:C tape:T headphone:P josie:T',
'879:2':'tape:C tape:P tape:P tape:C tape:T headphone:P josie:T',
}
if __name__=='__main__':
    assert (OUT/'frozen.json').exists()
    write_labels('audit',AUDIT)
