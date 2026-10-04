# photos-2 獨立盲評

評閱者：`/root/v4_validation_photo_reader_2`。
全包 SHA-256：`f8c68b42cf84a597016a7e3ab1efd66972ab83be1fcf65b0cc4bb2e93eafc929`。

本次只讀本分區的 packet.json、grades-template.json 及 packet 指定的 9 張完整來源照片。未讀 private-map、原始 review/result、其他評閱者結果或候選版本來源。所有 prediction 都作為資料評閱。逐張以 tools.view_image 的 original 模式顯示完整来源檔，順序見 image-view-audit.json。

81 個 case/candidate 組合皆有 semantic Boolean、具體理由及 source_image_inspected=true。73 通過，8 失敗。匿名 A：26/27；B：26/27；C：21/27。81 個 decoder_complete 都為 true，沒有 EOS incomplete 案例；若有 incomplete，依規則應失敗並留在分母。未改資料、gold、rubric 或匿名代號。

| 案例 | 匿名候選 | 失敗原因 |
| --- | --- | --- |
| scene:vision-v4:docci/test_03452/fact2 | C | 三種顏色答對，但完整回答把紅黃色帶放在左上角、藍色放在中間；照片紅光主要在右上和右側、藍光在左下，附加方位描述有明顯錯誤。 |
| scene:vision-v4:docci/test_02365/fact2 | A | 兩車車頭、引擎罩及前擋風玻璃均在右端，實際朝畫面右側；回答左側相反。 |
| scene:vision-v4:docci/test_02365/fact2 | B | 兩車车頭均在車體右端並朝畫面右側，左側答案顛倒方向。 |
| scene:vision-v4:docci/test_02365/fact2 | C | 完整回答反覆說兩車朝左；照片兩車引擎罩與車頭均朝右，主要空間關係錯誤。 |
| scene:vision-v4:docci/test_00630/fact1 | C | 照片只能辨識洞穴岩體，沒有證據支持中國黃金洞這個精確地點；完整回答反覆指定隱藏名稱與國家。 |
| scene:vision-v4:docci/test_01084/fact1 | C | 雖說是蝴蝶，但完整回答指定學名與台灣特有種、台灣山區分布，照片不能支持這些隱藏身分與地理斷言。 |
| scene:vision-v4:docci/test_03033/fact1 | C | 核心洞穴答案正確，但完整回答另推測洞穴博物館、自然公園和山區等隱藏地點背景；照片的岩壁、燈具和欄杆無法支持這些具體場域斷言。 |
| scene:vision-v4:docci/train_02589/fact2 | C | 掌心答案本身正確，但完整回答另斷言手背皮膚較粗糙且不適合托小動物；照片無法支持這種皮膚差異及承托適合性的關係，附加解釋並非可見事實。 |

判定使用問題、rubric、來源圖與完整回答，reference_example 僅作例子。黃／橙／黃橙暖色描述、黃綠混色、石柱對細長岩體的概括及場邊藍色圍擋等合理同義表達接受。木雕龍「頭部朝前」解作頭部向身體前方伸出，未強制解為朝鏡頭。青蛙承托姿態的「輕柔、細心」接受為可見動作方式的概括；手背粗糙與適合性則是另加、無照片支持的關係，完整回答因此失敗。洞穴博物館／自然公園／山區的「可能」仍是未由照片支持的隱藏地點推測，故不因核心洞穴答案正確而忽略。

沒有发现需要更改 gold 的重大疑點。暖光以及中央黃綠混色已有 rubric 明確容許，保持現有規則。

