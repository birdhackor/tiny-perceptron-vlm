# Fresh first-read notes before opening prior reviews

Reader: /root/natural_reader_mobile_11_14

Read 11.14 completely and direct prerequisites 11.4 and 10.2, with 10.1 for RGB axes and 11.3 for the description-to-question context. No old reader, technical review, or author plan has been opened.

My understanding: seeing objects alone does not establish their relationship. Swapping equally sized red and blue bars within one averaging cell retains each colour amount and hence the same mean, but the full ordered pixels change. Swapping across averaging cells changes their means. Retaining order is a necessary source of available information in this example, not proof of natural-photo reasoning. Natural photos require examples and checks for relationships and questions.

Paper exercise: with each original 8×8 cell, red and blue each occupy 4×8=32 positions, so the occupied first cell averages RGB [0.5,0,0.5], the second [0,0,0]. Move red to the second cell without changing bar size: the first becomes [0,0,0.5], the second [0.5,0,0], so the cell summaries differ. A photograph-supported question is “這個人的手是否正握著杯子？”; unsupported speculation is “他剛剛喝完咖啡，下一刻會離開。” Neither colour counts nor the picture alone supplies the unseen past/future.

128 prediction: 64 colour positions swap red and blue; each changes two channel entries, giving 64×2=128 scalar entries, not 128 coloured positions. The 4×4 pooled summary stays equal and the complete pixel sequence differs.
