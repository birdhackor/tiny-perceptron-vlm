# My independent reading notes for 20.4

The question is how to choose demonstrations that teach the intended ability before increasing the amount of data. An image plus the request and demonstrated answer teaches a relationship between those items. A folder label such as chat does not supply a useful answer.

I can check the cat example with the actual rendered image: there are two cats; both face the yellow monitor, and the left cat has a raised front paw. A one-to-two-sentence Traditional Chinese description teaches connecting visible objects and poses. A count question paired with the short answer two cats teaches responding directly to a quantity question. The photograph is introduced as a training demonstration, so I do not infer unseen-photo performance from this example. The section says its Chinese answers were written by the course AI assistant after viewing the photo and reading the original English annotation; I did not verify that annotation process.

The six source names label six uses. DOCCI supplies natural scenes and visible-fact questions; the synthetic document source pairs text regions with their transcriptions; Commons supplies photographed text; OpenAssistant supplies requests with answers that respect the conversation; FLEURS tests what was said; AISHELL separates transcribing a question from giving a response, with response criteria supplied by the course. These are claims presented by the prose, not data or benchmark results I reproduced. I did not open the optional acquisition list or any remote licensing/image link, and I downloaded no data.

The important numbers in the assigned section are two cats, a one-to-two-sentence answer, and making three copies of the same question-answer pair. I understand the last number as increased frequency/weight for one existing case, not three new situations. There are no formulas, formula symbols, fenced code blocks, or program outputs in 20.4.

My written exercise prediction and answer, without running a model:
- Choose a photographed shop sign or a synthetic text image, and pair the image with a request such as 請讀出招牌上的店名 and the actual visible name, for example 春光書店. The image and the answer must contain the same characters.
- Check the demonstration by comparing each answer character to the visible sign and confirming that the request asks for the name rather than a scene description.
- A classification answer 這是一間商店 is compatible with many different names. It gives no target character sequence and does not demonstrate how to read out the name.

The linked prerequisite 11.14 adds why an object/color label need not teach a spatial relation: red-left-of-blue and blue-left-of-red can have the same average. I read its explanation of 32-by-32 synthetic images, 8-by-8 averaging cells, a 4-by-4 summary, and the stated 128/True/False outputs; I did not execute its code. The linked prerequisite 12.13 distinguishes the copied dinner question from a helpful dinner suggestion. I read its stated 21 frames, 16 bands, 440/880 Hz tones, and False/True comparisons; I did not execute its code. Those numerical experiments are not needed to solve 20.4's exercise.

Two nonblocking vocabulary gaps remain: OCR is inferable from the paired-text explanation but its abbreviation is not expanded; same dialogue tree/same family is not locally defined. A reader can still solve the exercise and distinguish all three major tasks. A short parenthetical for each would make the data-use table more welcoming.

No code example or exercise was executed. The only execution was byte extraction/hashing, SVG rendering, and report/artifact persistence. This is a readability assessment, not a factual audit of dataset licensing or replication of an earlier experiment.
