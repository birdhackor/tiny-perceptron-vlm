# Independent derivation for 18.11

The constructed answer lists have three items and use equal item weights.
Identical teacher and student lists give agreement = 3/3 = 1.0.
The independent task rule checks two known arithmetic targets (2 and 4), and
accepts an admission of insufficient information for the third question.
The original list therefore passes 1/3 = 0.333333..., displayed as 0.3333.
Changing only the first teacher answer to the string "2" before copy passes
2/3 = 0.666666..., displayed as 0.6667. Agreement stays 3/3.
Rounding to four decimal places has maximum absolute rounding error 0.00005;
these examples have absolute error about 0.0000333333. The rates are unitless,
and the denominator is three questions, not answer tokens.

String exact match rejects "不知道" against "資訊不足". The explicit local
task rule accepts either expression at the third position. This is a constructed
demonstration of a criterion, not an automated semantic judge for open questions.

For a hard teacher token t, a student's cross entropy loss is
L = -log softmax(z)[t]. Its derivative is dL/dz[j] = p[j] - 1[j=t].
Consequently, gradient descent rewards the teacher token even if an external
task truth says it is wrong. This proves that an imitation objective supplies
no independent truth guarantee. It does not prove that a particular trained
student will repeat a teacher's exact string: optimization, initialization,
data coverage and the student's capacity also affect its outputs.

Parsing an object with one integer answer field establishes a format property.
It does not establish that the field equals the independent arithmetic target.
For 2+2 the target is 4; teacher 3 and student 10 both fail content despite
passing the saved JSON shape rule. Those are distinct wrong answers.

If the same arithmetic input is repeated under three style conditions, the
three observations are correlated views of that input. In the saved test data
there are 21 arithmetic rows: seven ordered (a,b) pairs, each under three
styles, belonging to four unordered arithmetic families. Six date rows belong
to three date families. These counts are not 27 independent novel arithmetic
problems. The program independently verifies disjoint family splits.

Logging each exclusion reason together with the source record/version permits
reconstruction of which signals the student was offered. This is an audit
property; it is not a theorem that filtering prevents all student errors.
