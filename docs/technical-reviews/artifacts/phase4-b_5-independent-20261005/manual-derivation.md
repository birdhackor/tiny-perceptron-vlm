B.5 hand-labelled task policy

1. The exact question asks for a numeric sum with operands 1 and 2. For available=True, the stipulated policy maps numerical_addition to TOOL; for available=False it maps to ASK. Operands do not change, so the latter asks for a verification route, not missing numbers.
2. Explain addition and copy the character 3 require no arithmetic operation under this policy: both are DIRECT. The occurrence of 加 or a digit cannot determine the action independently of intent.
3. 請算總價 supplies neither unit price nor quantity. A total p*q is not uniquely specified: e.g. p=2,q=3 gives 6 and p=4,q=3 gives 12. Thus ASK requests necessary information rather than a determinate numeric total.
4. There are five literal dict records: ordered action list TOOL,DIRECT,DIRECT,ASK,ASK. First and last question strings are byte-identical; only calculator state and the corresponding hand-label differ. This is five demonstration rows, not five predictions or an accuracy denominator.
5. A dict has independent calculator and action fields. Reassigning the former does not assign the latter. print reads stored values; it contains no decision branch, arithmetic evaluation, tool call or model update.

Scope: these are task-contract and literal Python checks, not a universal claim that easy arithmetic requires a calculator. No final numeric answer or trained capability is measured in B.5.
