<html>
<head>
    <title>ML 2026 Spring</title>
    <meta content="https://speech.ee.ntu.edu.tw/~hylee/images/ml-title-2022.png" property="og:image"/>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1, user-scalable=no" name="viewport"/>
    <link href="../assets/css/main.css" rel="stylesheet"/>
    <link crossorigin="anonymous" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css"
          integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
          integrity="sha512-9usAa10IRO0HhonpyAIVpjrylPvoDwiPUiKdWk5t3PyolY1cOd4DSE0Ga+ri4AuTroPR5aQvXU9xC6qOPnzFeg=="
          crossorigin="anonymous" referrerpolicy="no-referrer"/>
    <link href="../favicon.ico" rel="shortcut icon"/>
    <link href="../favicon.ico" rel="bookmark"/>
    <noscript>
        <link href="../assets/css/noscript.css" rel="stylesheet"/>
    </noscript>
    <style>
        p {
            font-size: 1.2rem;
        }

        * {
            box-sizing: border-box;
        }

        /* Set a background color */
        body {
            background-color: #4a412a;
        }

        a {
            color: #1a1241;
        }
    </style>
</head>

<body class="no-sidebar is-preload">

<!--

          _____                    _____                    _____                    _____                _____                    _____                            _____            _____                    _____
         /\    \                  /\    \                  /\    \                  /\    \              |\    \                  /\    \                          /\    \          /\    \                  /\    \
        /::\____\                /::\____\                /::\____\                /::\    \             |:\____\                /::\    \                        /::\____\        /::\    \                /::\    \
       /:::/    /               /:::/    /               /::::|   |               /::::\    \            |::|   |                \:::\    \                      /:::/    /       /::::\    \              /::::\    \
      /:::/    /               /:::/    /               /:::::|   |              /::::::\    \           |::|   |                 \:::\    \                    /:::/    /       /::::::\    \            /::::::\    \
     /:::/    /               /:::/    /               /::::::|   |             /:::/\:::\    \          |::|   |                  \:::\    \                  /:::/    /       /:::/\:::\    \          /:::/\:::\    \
    /:::/____/               /:::/    /               /:::/|::|   |            /:::/  \:::\    \         |::|   |                   \:::\    \                /:::/    /       /:::/__\:::\    \        /:::/__\:::\    \
   /::::\    \              /:::/    /               /:::/ |::|   |           /:::/    \:::\    \        |::|   |                   /::::\    \              /:::/    /       /::::\   \:::\    \      /::::\   \:::\    \
  /::::::\    \   _____    /:::/    /      _____    /:::/  |::|   | _____    /:::/    / \:::\    \       |::|___|______    ____    /::::::\    \            /:::/    /       /::::::\   \:::\    \    /::::::\   \:::\    \
 /:::/\:::\    \ /\    \  /:::/____/      /\    \  /:::/   |::|   |/\    \  /:::/    /   \:::\ ___\      /::::::::\    \  /\   \  /:::/\:::\    \          /:::/    /       /:::/\:::\   \:::\    \  /:::/\:::\   \:::\    \
/:::/  \:::\    /::\____\|:::|    /      /::\____\/:: /    |::|   /::\____\/:::/____/  ___\:::|    |    /::::::::::\____\/::\   \/:::/  \:::\____\        /:::/____/       /:::/__\:::\   \:::\____\/:::/__\:::\   \:::\____\
\::/    \:::\  /:::/    /|:::|____\     /:::/    /\::/    /|::|  /:::/    /\:::\    \ /\  /:::|____|   /:::/~~~~/~~      \:::\  /:::/    \::/    /        \:::\    \       \:::\   \:::\   \::/    /\:::\   \:::\   \::/    /
 \/____/ \:::\/:::/    /  \:::\    \   /:::/    /  \/____/ |::| /:::/    /  \:::\    /::\ \::/    /   /:::/    /          \:::\/:::/    / \/____/          \:::\    \       \:::\   \:::\   \/____/  \:::\   \:::\   \/____/
          \::::::/    /    \:::\    \ /:::/    /           |::|/:::/    /    \:::\   \:::\ \/____/   /:::/    /            \::::::/    /                    \:::\    \       \:::\   \:::\    \       \:::\   \:::\    \
           \::::/    /      \:::\    /:::/    /            |::::::/    /      \:::\   \:::\____\    /:::/    /              \::::/____/                      \:::\    \       \:::\   \:::\____\       \:::\   \:::\____\
           /:::/    /        \:::\__/:::/    /             |:::::/    /        \:::\  /:::/    /    \::/    /                \:::\    \                       \:::\    \       \:::\   \::/    /        \:::\   \::/    /
          /:::/    /          \::::::::/    /              |::::/    /          \:::\/:::/    /      \/____/                  \:::\    \                       \:::\    \       \:::\   \/____/          \:::\   \/____/
         /:::/    /            \::::::/    /               /:::/    /            \::::::/    /                                 \:::\    \                       \:::\    \       \:::\    \               \:::\    \
        /:::/    /              \::::/    /               /:::/    /              \::::/    /                                   \:::\____\                       \:::\____\       \:::\____\               \:::\____\
        \::/    /                \::/____/                \::/    /                \::/____/                                     \::/    /                        \::/    /        \::/    /                \::/    /
         \/____/                  ~~                       \/____/                                                                \/____/                          \/____/          \/____/                  \/____/

-->

<div id="page-wrapper">
    <!-- Header -->
    <header id="header">
        <h1 id="logo" style="font-size:medium; font-weight:500"><a href="../index.html">Hung-yi Lee (李宏毅)</a></h1>
        <nav id="nav">
            <ul>
                <li class="menu"><a href="/~hylee/index.php">Home</a></li><li class="menu"><a href="/~hylee/honor.php">Honor</a></li><li class="menu"><a href="/~hylee/research.php">Research</a></li><li class="menu"><a href="/~hylee/talk.php">Talk</a></li><li class="menu"><a href="/~hylee/publication.php">Publication</a></li><li class="submenu current">
			  <a href="#">Course</a>
                <ul>
                <li class="submenu">
                  <a href="#">GenAI-ML</a>
                  <ul>
                    <li><a href="/~hylee/GenAI-ML/2025-fall.php">2025 Fall</a></li>
                  </ul>
                </li>
                <li class="submenu">
                  <a href="#">Generative AI</a>
                  <ul>
                    <li><a href="/~hylee/genai/2024-spring.php">2024 Spring</a></li> 
                  </ul>
                </li>
                  <li class="submenu">
                    <a href="#">Machine Learning</a>
					<ul>
					  <li><a href="/~hylee/ml/2026-spring.php">2026 Spring</a></li>
					  <li><a href="/~hylee/ml/2025-spring.php">2025 Spring</a></li>
				      <li><a href="/~hylee/ml/2023-spring.php">2023 Spring</a></li> 
                      <li><a href="/~hylee/ml/2022-spring.php">2022 Spring</a></li>
                      <li><a href="/~hylee/ml/2021-spring.php">2021 Spring</a></li>
                      <li><a href="/~hylee/ml/2020-spring.php">2020 Spring</a></li>
                      <li><a href="/~hylee/ml/2019-spring.php">2019 Spring</a></li>
                      <li><a href="/~hylee/ml/2017-fall.php">2017 Fall</a></li>
                      <li><a href="/~hylee/ml/2017-spring.php">2017 Spring</a></li>
                      <li><a href="/~hylee/ml/2016-fall.php">2016 Fall</a></li>
                    </ul>
                  </li>
                  <li class="submenu">
                    <a href=#>DLHLP</a>
                    <ul>
                      <li><a href="/~hylee/dlhlp/2020-spring.php">2020 Spring</a></li>
                    </ul>
                  </li>
                  <li class="submenu">
                    <a href="#">MLDS</a>
                    <ul>
                      <li><a href="/~hylee/mlds/2018-spring.php">2018 Spring</a></li>
                      <li><a href="/~hylee/mlds/2017-spring.php">2017 Spring</a></li>
                      <li><a href="/~hylee/mlds/2015-fall.php">2015 Fall</a></li>
                    </ul>
                  </li>
                  <li class="submenu">
                    <a href="#">Linear Algebra</a>
                    <ul>
                      <li><a href="https://voidful.github.io/LA_2023_fall/2023-fall.html">2023 Fall</a></li>
                      <li><a href="https://googly-mingto.github.io/LA_2022_fall/2022-fall.html">2022 Fall</a></li>
                      <li><a href="/~hylee/la/2021-fall.php">2021 Fall</a></li>
                      <li><a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses/LA_2020/policy.pdf" target="_blank" rel="noreferrer noopener">2020 Fall</a></li>
                      <li><a href="/~hylee/la/2019-fall.php">2019 Fall</a></li>
                      <li><a href="/~hylee/la/2018-fall.php">2018 Fall</a></li>
                      <li><a href="/~hylee/la/2016-spring.php">2016 Spring</a></li>
                    </ul>
                  </li>
                  <li><a href="/~hylee/circuit/2014-fall.php">Circuit</a></li>
                </ul>
              </li>
              <li class="menu"><a href="https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists" target="_blank" rel="noopener noreferrer">Youtube</a></li>             </ul>
        </nav>
    </header>
    <!-- Main -->
    <article id="main">
        <header class="special container"><span class="icon solid fas fa-laptop-code" style="color:#ffffff"></span>
            <h2 style="color:white">Machine Learning 2026 Spring</h2>
            <br>
            <h3 id="cd" style="color:white"></h3></header>
        <script>
            // Set the date we're counting down to
            var countDownDate = new Date("Feb 18, 2022 14:20:00").getTime();
            // Update the count down every 1 second
            var x = setInterval(function () {
                // Get today's date and time
                var now = new Date().getTime();
                // Find the distance between now and the count down date
                var distance = countDownDate - now;
                // Time calculations for days, hours, minutes and seconds
                var days = Math.floor(distance / (1000 * 60 * 60 * 24));
                var hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
                var minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
                var seconds = Math.floor((distance % (1000 * 60)) / 1000);
                // Output the result in an element with id="demo"
                document.getElementById("cd").innerHTML = "Course starts in " + days + "d " + hours + "h " + minutes + "m " + seconds + "s";
                // If the count down is over, write some text
                if (distance < 0) {
                    clearInterval(x);
                    document.getElementById("cd").innerHTML = "";
                }
            }, 1000);
        </script>
        <section class="wrapper style4 container" style="border-radius:20px;">
            <div class="content">
                <section>
                    <header style="text-align:center">
                        <h3>News</h3></header>
                </section>
            </div>
            
            <div class="row row-striped" style="border:5px solid #e0e0eb; border-radius:20px; margin-bottom:10px">
                <div class="col-2 text-right" style="text-align:center; margin-right:0">
                    <h1 class="display-4"><span class="badge bg-danger">1</span></h1>
                    <h2>Jun</h2></div>
                <div class="col-1"></div>
                <div class="col-9">
                    <h3 class="text-uppercase"><strong>作業十公布</strong></h3>
                    <p style="font-size:1.2rem">詳細已公告於ntucool以及下方課程內容和作業區
                </div>
            </div>
            
            
            <div class="row row-striped" style="border:5px solid #e0e0eb; border-radius:20px; margin-bottom:10px">
                <div class="col-2 text-right" style="text-align:center; margin-right:0">
                    <h1 class="display-4"><span class="badge bg-danger">7</span></h1>
                    <h2>Mar</h2></div>
                <div class="col-1"></div>
                <div class="col-9">
                    <h3 class="text-uppercase"><strong>課程加簽碼已全數寄出</strong></h3>
                    <p style="font-size:1.2rem">課程加簽碼已全數寄出，再請同學查詢信箱
                </div>
            </div>
           
            
            

  
          
        </section>
        <section class="wrapper style4 container" style="border-radius:20px;">
            <!-- Content -->
            <div class="content">
                <section class="row row-striped" style="border-radius:20px;">
                    <!-- SYLLABUS -->
                    <div class="content" id="syllabus">
                        <section>
                            <header style="text-align:center">
                                <h3>Content</h3></header>
                                 <br>
						        </br>
                                <div class="content" id="syllabus">
			                 
                            <div style="overflow-x:auto">
                                <table class="table">
                                    <thead>
                                    <tr>
                                        <th><span>Date</span></th>
                                        <th><span>Topic</span></th>
                                        <th><span>Class Material</span>
                                        </th>
                                        <th><span>Extra Material</span>
                                        </th>
                                        <th><span>Optional Material</span>
                                        </th>
                                        </th>
                                        <th><span>AI generated video</span>
                                        </th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    <tr>
                                        <td><span>3/6</span></td>
                                        <td><span>AI Agent - 1</span>
                                          
                                        </td>
                                        <td>
                                        
                                            <!--<span>機器學習2026規則說明 </span> -->
                                            <i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/gl-BdDjNPVI">機器學習2026規則說明</a> /  
                                            <i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/policy.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/policy.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                                
	                                        <br>
	                                        <!--<span>解剖小龍蝦 — 以 OpenClaw 為例介紹 AI Agent 的運作原理</span>-->
	                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/2rcJdFuNbZQ">解剖小龍蝦 — 以 OpenClaw 為例介紹 AI Agent 的運作原理</a> /  
	                                        	<i class="fa-regular fa-file-powerpoint"></i>
	                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/intro.pptx"
	                                               rel="noopener noreferrer" target="_blank">ppt</a> /
	                                            <i class="fa-regular fa-file-pdf"></i><a
	                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/intro.pdf"
	                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                            </br>
                                            
                                            
                                        </td>
                                        <td>
                                        		<!--<span>【生成式人工智慧與機器學習導論2025】第１講：一堂課搞懂生成式人工智慧的原理</span>-->
	                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/TigfpYPJk1s?si=Vq73vVihiP67EYdZ">【生成式人工智慧與機器學習導論2025】第１講：一堂課搞懂生成式人工智慧的原理</a>  
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>3/13</span></td>
                                        <td><span>AI Agent - 2</span>
                                            
                                        </td>
                                        <td>
                                        
                                            <!--<span>AI Agent 的技術發展 </span> -->
                                            <i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/urwDLyNa9FU">Context Engineering 基本概念解說 /  
                                            <br>
                                            <i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/mmPmNezjCi0">AI Agent 之間可以有什麼樣的互動/  
                                            <br>
                                            <i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/VqB8zMujdjM">AI Agent 對於工作帶來的衝擊 - 以學術研究為例/  
                                            <br>
                                            <i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/agent_era.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/agent_era.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                                
	                             
                                        </td>
                                        <td>
                                        		<!--<span>【生成式人工智慧與機器學習導論2025】第１講：一堂課搞懂生成式人工智慧的原理</span>-->
	                                        	<!--<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/TigfpYPJk1s?si=Vq73vVihiP67EYdZ">【生成式人工智慧與機器學習導論2025】第１講：一堂課搞懂生成式人工智慧的原理</a>  -->
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>3/20</span></td>
                                        <td><span>深入模型內部架構：如何加快模型推論速度</span>
                                            
                                        </td>
                                        <td>
                                         <!--<span>加快語言模型的生成速度 </span> -->
                                     
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/vXb2QYOUzl4">加快語言模型生成速度 (1/2)：Flash Attention</a> 
                                        	<br>
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/fDQaadKysSA">加快語言模型生成速度 (2/2)：KV Cache</a>/ 
                                        	<br>
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/inference.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/inference.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>3/27</span></td>
                                        <td><span>深入模型內部架構：模型如處理超長輸入</span>
                                            
                                        </td>
                                        <td>
                                        	<!--<span>Positional Embedding</span>-->
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/Ll-wk8x3G_g">Positional Embedding</a> /  
                                        	 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/pos.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/pos.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        <td>
                                        </td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>4/10</span></td>
                                        <td><span>如何教育模型 - 1</span>
                                            
                                        </td>
                                        <td>
                                        	<!--<span>Harness Engineering </span>-->
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/R6fZR_9kmIw">Harness Engineering</a> /
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/harness.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/harness.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        <!--<span>教育部AI cup 說明會（蔡宗翰老師）</span>-->
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>4/24</span></td>
                                        <td><span>如何教育模型 - 2</span>
                                            
                                        </td>
                                        <td>
                                        	
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/m3i2mk5hs8U">Self-Correction</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/Self-Correction.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/Self-Correction.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        	<!--<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/mpuRca2UZtI">如何在多張GPU上訓練大型模型（王秀軒助教授課)</a> -->

                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>5/8</span></td>
                                        <td><span>模型的自我成長 - 1</span>
                                            
                                        </td>
                                        <td>
                                        	
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/s06mSAGN4gM">Self-Improving</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/Self-Improving.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/Self-Improving.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        	<!--<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/mpuRca2UZtI">如何在多張GPU上訓練大型模型（王秀軒助教授課)</a> -->

                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>5/15</span></td>
                                        <td><span>Appier Research 團隊演講</span>
                                        </td>
                                       <td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>5/22</span></td>
                                        <td><span>模型的自我成長 - 2</span>
                                            
                                        </td>
                                        <td>
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/cQLKVzbwN7I">Self-Improving -2</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/self-evolving-agent.ptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/self-evolving-agent.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        	<!--<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/mpuRca2UZtI">如何在多張GPU上訓練大型模型（王秀軒助教授課)</a> -->

                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>5/29</span></td>
                                        <td><span>Spoken LM TALK</span>
                                        </td>
                                        <td>
                                        <span> 楊書文、楊智凱同學演講</span>
                                        	<!--<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/bJFtcwLSNxI">Reasoning</a> -->
                                        	<!--<i class="fa-regular fa-file-powerpoint"></i>-->
                                         <!--   <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/reasoning.pptx"-->
                                         <!--      rel="noopener noreferrer" target="_blank">ppt</a> /-->
                                         <!--   <i class="fa-regular fa-file-pdf"></i><a-->
                                         <!--       href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/reasoning.pdf"-->
                                         <!--       rel="noopener noreferrer" target="_blank"> pdf</a>-->
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>6/05</span></td>
                                        <td><span>陳暐教授演講</span>
                                        </td>
                                        <td>
                                        	<!--<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/bJFtcwLSNxI">Reasoning</a> -->
                                        	<!--<i class="fa-regular fa-file-powerpoint"></i>-->
                                         <!--   <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/reasoning.pptx"-->
                                         <!--      rel="noopener noreferrer" target="_blank">ppt</a> /-->
                                         <!--   <i class="fa-regular fa-file-pdf"></i><a-->
                                         <!--       href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/reasoning.pdf"-->
                                         <!--       rel="noopener noreferrer" target="_blank"> pdf</a>-->
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    </tbody>
                                </table>
                            </div>
                        </section>
                    </div>
                </section>
                <section class="row row-striped" style="border-radius:20px;">
                    <!-- Content -->
                    <div class="content" id="hw">
                        <section>
                            <header style="text-align:center">
                                <h3>Homework</h3></header>
                            <div>
                                <p style="font-size:large; text-align:left"> The information here is tentative and
                                    subject to change. Please read the requirement of each homework before deadline.
                                <div class="alert alert-danger" role="alert" style="text-align:left; font-size:large">
                                    <ul>
                                        <li>You should NOT plagiarize, if you use any other resource, you should cite it in the reference.</li>
                                        <li>You should NOT modify your prediction files manually.</li>
                                        <li>Do NOT share codes or prediction files with any living creatures.</li>
                                        <li>Your final grade x 0.9 and get a score 0 for that homework if you violate any of the above rules first time (within a semester)
                                        </li>
                                        <li>Your will get F for the final grade if you violate any of the above rules multiple times (within a semester).
                                        </li>
                                        <li>Prof. Lee & TAs preserve the rights to change the rules & grades.</li>
                                    </ul>
                                </div>
                                </p>
                            </div>
                            <div style="overflow-x:auto">
                                <table class="table">
                                    <thead>
                                    <tr>
                                        <th><span>#</span></th>
                                        <th><span>Date</span></th>
                                        <th><span>Topic</span>
                                        </th>
                                        <th><span>Video</span>
                                        </th>
                                        <th><span>Slides</span>
                                        </th>
                                        <th><span>Code</span></th>
                                        <th><span>Platform</span>
                                        </th>
                                        <th><span>Deadline (UTC+8)</span>
                                        </th>
                                        <th><span>TA</span>
                                        </th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    <tr>
                                        <td><span>x</span></td>
                                        <td><span>3/6</span></td>
                                        <td><span>Colab and Kaggle Tutorial</span>
                                        </td>
                                        <td>
                                        	<a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://www.youtube.com/watch?v=kibL4oJbzy4">
                                            </a>
                                        </td>
                                        <td><span>
                                            <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/Colab_Tutorial.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td><span>
                                        	<a class="fa-brands fa-python fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/14CEwML9XSpxSvZoHfvZTB1h8ZWC8CYyT?usp=sharing">
                                            </a>
                                            <a class="fa-brands fa-kaggle fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/walkerhsu/kaggle-tutorial">
                                            </a>
                                        </span>
                                        <td><span>N/A</span>
                                        </td>
                                        <td><span>N/A</span>
                                        </td>
                                        <td><span>N/A</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>x</span></td>
                                        <td><span>3/6</span></td>
                                        <td><span>PyTorch Tutorial</span>
                                        </td>
                                        <td><span>
                                            <a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://youtu.be/6dEp6oRN2NE">
                                            </a>
                                        </span>
                                        </td>
                                        <td><span>
                                            <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/Pytorch_Tutorial_1_rev_1.pdf">
                                            </a>
                                            <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/Pytorch_Tutorial_2.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td><span>N/A</a></span>
                                        </td>
                                        <td><span>N/A</span>
                                        </td>
                                        <td><span>N/A</span>
                                        </td>
                                        <td><span>N/A</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>x</span></td>
                                        <td><span>3/6</span></td>
                                        <td><span>JudgeBoi Guide</span>
                                        </td>
                                        <td>
                                        	<a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://www.youtube.com/watch?v=sNX3iKzxAPs">
                                            </a>
                                        </td>
                                        <td><span>
                                            <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/JudgeBoi_Guide.pdf">
                                            </a>
                                           
                                        </span>
                                        </td>
                                        <td><span>N/A</a></span>
                                        </td>
                                        <td><span>N/A</span>
                                        </td>
                                        <td><span>N/A</span>
                                        </td>
                                        <td><span>N/A</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>Bonus</span></td>
                                        <td><span>3/13</span></td>
                                        <td><span>Bonus Competition</span>
                                        </td>
                                        <td><span>
                                            <a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://youtu.be/Xk_utjmoK5A">
                                            </a>
                                        </span>
                                        </td>
                                        <td><span>
                                            <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/bonus.pdf">
                                            </a>
                                            
                                        </span>
                                        </td>
                                        <td><span>N/A</a></span>
                                        </td>
                                        <td><span>N/A</a></span>
                                        	<!--<span>-->
                                        	<!--<a class="fa-brands fa-kaggle fa-xl"-->
                                         <!--      target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/t/bf7880c9aa95483580e0c7a500b122b2">-->
                                         <!--   </a>-->
                                        	<!--<a class="fa-brands fa-kaggle fa-xl"-->
                                         <!--      target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/competitions/ml-2026-bonus-competition/overview">-->
                                         <!--   </a>-->
                                        	<!--</span>-->
                                        </td>
                                        <td><span>05/15/2026 19:59</span>
                                        </td>
                                        <td><span>許筠曼</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 1</span></td>
                                        <td><span>3/6</span></td>
                                        <td>
                                        <span>LLM Malicious Instruction Defense</span>
                                        </td>
                                        <td>
                                        	<a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://youtu.be/qVTehGJQHys">
                                            </a>
                                        </td>
                                        <td>
                                        <span>
                                            <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/hw1.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td>
                                        <span><a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1kgy1Nas2uu7RoWTn-pZfyF1Sj3WpuJdy?usp=sharing"></a></span>
                                                
                                                <!--<span><a class="fa-brands fa-kaggle fa-xl"-->
                                                <!--     target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/u0ulin/ml2026-homework-1"></a></span>-->
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/homeworks/1/description">
                                            </a>
                                        </td>
                                        <td>
                                        	<span>03/26/2026 23:59</span>
                                        </td>
                                        <td>
                                        	<span>董家愷、陳思齊、許筠曼</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 2</span></td>
                                        <td><span>3/13</span></td>
                                        <td>
                                        <span>AI Agent as an AI Engineer</span>
                                        </td>
                                        <td>
                                        	<a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://youtu.be/3xhwSsuNTM0">
                                            </a>
                                        </td>
                                        <td>
                                        <span>
                                            <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/hw2.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td>
                                        <span><a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1hAT97f4GmBQFpWKHiRymIDJiXEsPlXS1?usp=sharing"></a></span>
                                                
                                        <!--        <span><a class="fa-brands fa-kaggle fa-xl"-->
                                        <!--             target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/yuxianglin032/ml2026spring-hw2-public"></a></span>-->
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/homeworks/2/description">
                                            </a>
                                        </td>
                                        <td>
                                        	<span>04/02/2026 23:59</span>
                                        </td>
                                        <td>
                                        	<span>林禹融 劉建蘴 楊樂霖</span>
                                        </td>
                                    </tr>
                                    <tr>
                                    </tr>
                                    <tr>
                                        <td><span>HW 3</span></td>
                                        <td><span>3/20</span></td>
                                        <td>
                                        <span>LLM Fast Inference</span>
                                        </td>
                                        <td>
                                            <a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://youtu.be/rXfp9Yo5HwU?si=55cf_qFbJlK5tgiB">
                                            </a>
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/hw3.pdf">
                                            </a>
                                            
                                        </span>
                                        </td>
	                                        <td>
		                                        <span><a class="fa-brands fa-python fa-xl"
		                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1vZNo6_PlaP2fvMqr3g5KoQA0rN79m24O?usp=sharing"></a></span>
		                                                
	                                        </td>
	                                        <td>
	                                        <span> NTUCOOL </span>
		                                        <!--<a class="fa-solid fa-g fa-xl"-->
	                                         <!--       target="_blank" rel="noopener noreferrer" href="https://www.gradescope.com/login">-->
	                                            <!--</a>-->
	                                      
	                                        </td>
	                                        <td>
	                                        	<span>04/09/2026 23:59</span>
	                                        </td>
	                                        <td>
	                                        	<span>馮柏翰, 吳岳霖, 蘇炳揚</span>
	                                        </td>
	                                	</td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 4</span></td>
                                        <td><span>3/27</span></td>
                                        <td>
                                        <span>Training Transformer</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/QrqdoGf35Iw">
                                            </a>
                                        	
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data//hw4.pdf">
                                            </a>
                                            
                                        </span>
                                        </td>
	                                        <td>
	                                        <span><a class="fa-brands fa-python fa-xl"
	                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1G9CgvnhqQ5AwHc6nbVzVGCoe-xUXdSWB?usp=sharing"></a></span>
                                            
                                            <span><a class="fa-brands fa-kaggle fa-xl"
                                                 target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/stevenlunar/ml2026-spring-hw4-training-transformer"></a></span>
                                        </td>
	                                        <td>
		                                        <a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/home">
                                            	</a>
	                                        </td>
	                                        <td>
	                                        	<span>04/16/2026 23:59</span>
	                                        </td>
	                                        <td>
	                                        	<span>劉建蘴、馮柏翰、陳品睿</span>
	                                        </td>
	                                	</td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 5</span></td>
                                        <td><span>4/10</span></td>
                                        <td>
                                        <span>Finetuning without Forgetting</span>
                                        </td>
                                    
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/HlSGih7bnrs">
                                            </a>
                                        	
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data//hw5.pdf">
                                            </a>
                                            
                                        </span>
                                        </td>
                                        <td>
                                        <span><a class="fa-brands fa-python fa-xl"
			                                     target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1H5FZA-l5n7QD1Q8vnBEUSlldVKlpchku#scrollTo=S3jAr39UAp8B"></a></span>
		                                <span><a class="fa-brands fa-kaggle fa-xl"
                                                 target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/b10901024sillydinos/ml2026hw5/edit/run/306310732"></a></span>            
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/homeworks/4/description">
                                            	</a>
                                        </td>
                                        <td>
                                        	<span>04/30/2026 23:59</span>
                                        </td>
                                        <td>
                                        	<span>謝翔、尹廷安、蘇炳揚</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 6</span></td>
                                        <td><span>4/24</span></td>
                                        <td>
                                        <span>Model Editing</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/AR1bNACLOAU?si=oaVtkj6YKEtcVWhG">
                                           </a>
                                           <!--<a class="fa-brands fa-youtube-square fa-xl"-->
                                           <!--target="_blank" rel="noopener noreferrer" href="https://youtu.be/b8fad34gpFY">-->
                                           <!--</a> -->
                                        	
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/hw6.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td>
                                		    <span><a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1gnaowsSzOT3VSw8j_MIDnksQiaZeKikA?usp=sharing"></a></span>
                                                
                              
                                        </td>
                                        <td>
                                        	<span> NTUCOOL</span>
                                        </td>
                                        <td>
                                        	<span>05/14/2026 23:59</span>
                                        </td>
                                        <td>
                                        	<span>鄭安妤、楊樂霖、尹廷安、林育正</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 7</span></td>
                                        <td><span>5/8</span></td>
                                        <td>
                                        <span>Model Merging</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/YQtwk_L686I">
                                           </a>	
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/hw7.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td>
                                		    <span><a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1B9692EHFAZFh5-8Q5LsVhzk9nTH1MEyD"></a></span>
                                            <span><a class="fa-brands fa-kaggle fa-xl"
                                             target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/sylora1101/ml2026hw7"></a></span>  
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/home">
                                            	</a>
                                            
                                        </td>
                                        <td>
                                        	<span>05/28/2026 23:59</span>
                                        </td>
                                        <td>
                                        	<span>黃郁涵、陳思齊、董家愷、吳岳霖</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 8</span></td>
                                        <td><span>5/15</span></td>
                                        <td>
                                        <span>Test-Time Scaling</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/KAbM5gM6Isw">
                                           </a>	
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/hw8.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td>
                                		    <span><a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1_z4JryPWnITLAwtytVwu75FZMx9giT3R?usp=sharing"></a></span>
                                                
                                        </td>
                                        <td>
                                        	<span>NTUCOOL</span>
                                        	<!--<a class="fa-solid fa-g fa-xl"-->
                                         <!--      target="_blank" rel="noopener noreferrer" href="https://www.gradescope.com/courses/1000456">-->
                                         <!--   	</a>-->
                                        </td>
                                        <td>
                                        	<span>06/04/2026 23:59</span>
                                        </td>
                                        <td>
                                        	<span>江履方、陳品睿、尹廷安、林育正</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 9</span></td>
                                        <td><span>5/22</span></td>
                                        <td>
                                        <span>Flow Matching</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/wAAeuMQ9r5c?si=55CubyBtqTZj6bH6">
                                           </a>	
                                        </td>
                                        <td>
                                        <span>
                                        	<a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/hw9.pdf">
                                            </a>
                                        </span>
                                        
                                        </td>
                                        <td>
                                		    <span>
                                		    	<a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1R1CNujj6-kVPkl53RQLt5kE7tYVS-Zmp?usp=sharing">
                                                </a>
                                            </span>
                                            
                                            <!--<span>-->
                                            <!--	<a class="fa-brands fa-kaggle fa-xl"-->
                                            <!--    target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/monicahuangml1/ml2026-hw9-model-merging">-->
                                            <!--    </a>-->
                                            <!--</span>-->
                                                
                                        </td>
                                        <td>
                                        <span>NTUCOOL</span>
                                        	<!--<a class="fa-solid fa-j fa-xl"-->
                                         <!--      target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/home">-->
                                         <!--   	</a>-->
                                        </td>
                                        <td>
                                        	<span>06/11/2026 23:59</span>
                                        </td>
                                        <td>
                                        	<span>林育正、吳岳霖、林禹融、<br>蘇炳揚、陳品睿、江履方</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 10</span></td>
                                        <td><span>5/29</span></td>
                                        <td>
                                        <span>Spoken Language Model</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/Gx96VH6ePC4">
                                           </a>	
                                        </td>
                                        <td>
                                        <span>
                                        	<a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2026-course-data/hw10.pdf">
                                            </a>
                                        </span>
                                        
                                        </td>
                                        <td>
                                		    <span>
                                		    	<a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1QBtp0lQrjQbTKB1sLIxoavqhSU7EhG_g?usp=sharing">
                                                </a>
                                            </span>
                                        </td>
                                        <td>
                                        	<span>NTUCOOL</span>
                                        	<!--<a class="fa-solid fa-j fa-xl"-->
                                         <!--      target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/home">-->
                                         <!--   	</a>-->
                                        </td>
                                        <td>
                                        	<span>06/18/2026 23:59</span>
                                        </td>
                                        <td>
                                        	<span>陳竣瑋、陳思齊、鄭安妤、尹廷安</span>
                                        </td>
                                    </tr>
                                    </tbody>
                                </table>
                            </div>
                        </section>
                    </div>
                </section>
                <div class="like_button_container" data-commentid="3"></div>
                <section class="wrapper style4 container" style="border-radius:20px;">
                    <!-- Content -->
                    <div class="content">
                        <section>
                            <header style="text-align:center">
                                <h3>Contact</h3></header>
                            <p style="font-size:large; text-align:center"> Email: <a
                                    href="mailto:ntu-ml-2026-spring-ta@googlegroups.com">ntu-ml-2026-spring-ta@googlegroups.com</a>
                            </p>
                            <p style="font-size:medium; text-align:center">
                                * 請同學來信務必依照正確寄信格式，並請善用NTU Cool討論區
                            </p>
                        </section>
                    </div>
                </section>
            </div>


        </section>
    </article>
    <!--voidful-->
    <!-- Footer -->
    <footer id="footer">
        <!--        <pre style="font-size: x-small">-->
        <!-- __  __            _     _              _                           _-->
        <!--|  \/  |          | |   (_)            | |                         (_)-->
        <!--| \  / | __ _  ___| |__  _ _ __   ___  | |     ___  __ _ _ __ _ __  _ _ __   __ _-->
        <!--| |\/| |/ _` |/ __| '_ \| | '_ \ / _ \ | |    / _ \/ _` | '__| '_ \| | '_ \ / _` |-->
        <!--| |  | | (_| | (__| | | | | | | |  __/ | |___|  __/ (_| | |  | | | | | | | | (_| |-->
        <!--|_|  |_|\__,_|\___|_| |_|_|_| |_|\___| |______\___|\__,_|_|  |_| |_|_|_| |_|\__, |-->
        <!--       _             _    _                                _   _          __/ |-->
        <!--      | |           | |  | |                              (_) | |        |___/-->
        <!--      | |__  _   _  | |__| |_   _ _ __   __ _ ______ _   _ _  | |     ___  ___-->
        <!--      | '_ \| | | | |  __  | | | | '_ \ / _` |______| | | | | | |    / _ \/ _ \-->
        <!--      | |_) | |_| | | |  | | |_| | | | | (_| |      | |_| | | | |___|  __/  __/-->
        <!--      |_.__/ \__, | |_|  |_|\__,_|_| |_|\__, |       \__, |_| |______\___|\___|-->
        <!--              __/ |                      __/ |        __/ |-->
        <!--             |___/                      |___/        |___/-->

        <!--        </pre>-->

        <ul class="icons">
            <li><a class="icon brands fa-facebook-f"
                   href="https://www.facebook.com/profile.php?id=100000149111577"><span
                    class="label">Facebook</span></a></li>
            <li><a class="icon brands fa-youtube"
                   href="https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists"><span
                    class="label">Youtube</span></a></li>
        </ul>
        <ul class="copyright">
            <li>&copy; Untitled</li>
            <li>Design:<a href="http://html5up.net">HTML5 UP</a></li>
        </ul>
    </footer>
</div>
<!-- Scripts -->
<script crossorigin="anonymous" integrity="sha384-ygbV9kiqUc6oa4msXn9868pTtWMgiQaeYH7/t7LECLbyPA2x65Kgf80OJFdroafW"
        src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/js/bootstrap.bundle.min.js"></script>
<script src="../assets/js/jquery.min.js"></script>
<script src="../assets/js/jquery.dropotron.min.js"></script>
<script src="../assets/js/jquery.scrolly.min.js"></script>
<script src="../assets/js/jquery.scrollex.min.js"></script>
<script src="../assets/js/browser.min.js"></script>
<script src="../assets/js/breakpoints.min.js"></script>
<script src="../assets/js/util.js"></script>
<script src="../assets/js/main.js"></script>
<script>
    const str = `
((((&&&&&&&&&&&%&&&&&&&&&&&&&&&&&&&&%%#%##%%%#####
((((&&&&&&&&&%&&&&   .    &&&&&&&&&%%%%%%%%%##%###
((##&&&&&&&%%%&@@     ,*., &&&&&&&&&&&&&&%%%%%%%%#
####&&&&&&&&&&&@@ ,./#(*(/*&&&&&&&&%%&&&&###%%%%%%
####%&@@@@@@@@@@@@ **//,.*&&@&&&&%&&#%%&&&%%%%%%%%
####%&@@@@@@@&@&&@.*,....((@@&&&&&&&&&&&&%%%%%%%%#
#####&@@@@@@@@*,,,,  ,.. ,*.(@&&&&&&&&&&&&&%%#%%%#
#####@@@@@@,..,,,,,,,.  ,/###,,&&&&&&&&&&&&%%%%%%%
#####@@@@@......,,,,,/..,.(*,*.,&&&&&&&&&&&&%%%%%%
#####@@@@..... ...,,,,,.... ..,,,&&&&&&&&&&&%%%%%%
#####&@@,,.        ... ...    ..*/&&&&&&&&&&%%%%%%
#####&&&,((#&     ...#%#(..    ...&&&&&&&&&%%%%%%%
#####&&&...,,,**//*(#  %&...  &&&&&&&&&&&&%%%%%%%%
#####&&&&&&&&       .(&&%&....&&&&&&&&&&&%%%%%%%%#
`;
    console.log(str)
</script>
</body>

</html>
