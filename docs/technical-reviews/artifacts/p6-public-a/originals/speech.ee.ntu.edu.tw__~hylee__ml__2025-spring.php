<html>
<head>
    <title>ML 2025 Spring</title>
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
            background-color: #5c46ab;
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
            <h2 style="color:white">Machine Learning 2025 Spring</h2>
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
                    <h1 class="display-4"><span class="badge bg-danger">20</span></h1>
                    <h2>May</h2></div>
                <div class="col-1"></div>
                <div class="col-9">
                    <h3 class="text-uppercase"><strong>Nvidia 演講上傳</strong></h3>
                    <p style="font-size:1.2rem">演講已上傳至ntucool week 13
                </div>
            </div>
            
            
            <div class="row row-striped" style="border:5px solid #e0e0eb; border-radius:20px; margin-bottom:10px">
                <div class="col-2 text-right" style="text-align:center; margin-right:0">
                    <h1 class="display-4"><span class="badge bg-danger">10</span></h1>
                    <h2>May</h2></div>
                <div class="col-1"></div>
                <div class="col-9">
                    <h3 class="text-uppercase"><strong>Bonus competition 期限延長</strong></h3>
                    <p style="font-size:1.2rem">Bonus competition已延長一週，請於下方作業區查看。
                </div>
            </div>
            
            
            <div class="row row-striped" style="border:5px solid #e0e0eb; border-radius:20px; margin-bottom:10px">
                <div class="col-2 text-right" style="text-align:center; margin-right:0">
                    <h1 class="display-4"><span class="badge bg-danger">23</span></h1>
                    <h2>Mar</h2></div>
                <div class="col-1"></div>
                <div class="col-9">
                    <h3 class="text-uppercase"><strong>加分作業公布</strong></h3>
                    <p style="font-size:1.2rem">詳細已公告於ntucool課程內容之加分作業區
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
                                    </tr>
                                    </thead>
                                    <tbody>
                                    <tr>
                                        <td><span>2/21</span></td>
                                        <td><span></span>
                                            一堂課搞懂生成式AI 
                                        </td>
                                        <td>
                                        
                                            <span>機器學習2025規則說明 </span> 
                                        
                                            <i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/policy.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/policy.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                                
	                                        <br>
	                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/QLiKmca4kzI">機器學習2025課程介紹</a> /  
	                                        	<i class="fa-regular fa-file-powerpoint"></i>
	                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/introduction.pptx"
	                                               rel="noopener noreferrer" target="_blank">ppt</a> /
	                                            <i class="fa-regular fa-file-pdf"></i><a
	                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/introduction.pdf"
	                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                            </br>
                                            
                                            
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>2/28</span></td>
                                        <td><span></span>
                                            國定假日
                                        </td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>3/7</span></td>
                                        <td><span></span>
                                            AI Agent 
                                        </td>
                                        <td>
                                     
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/M2Yg1kwPpts">AI Agent</a> /  
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/ai_agent.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/ai_agent.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>3/14</span></td>
                                        <td><span></span>
                                            剖析大型語言模型內部運作邏輯
                                        </td>
                                        <td>
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/Xnil63UDW2o">model inside</a> /  
                                        	 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/model_inside.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/model_inside.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        	
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>3/21</span></td>
                                        <td><span></span>
                                            Mamba  
                                        </td>
                                        <td>
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/gjsdVi90yQo">Mamba</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/mamba.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/mamba.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        <span>教育部AI cup 說明會（蔡宗翰老師）</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>3/28</span></td>
                                        <td><span></span>
                                            大型語言模型的神奇能力哪裡來 
                                        </td>
                                        <td>
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/Ozos6M1JtIE">Pretrain Alignment</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/pretrain.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/pretrain.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/mpuRca2UZtI">如何在多張GPU上訓練大型模型（王秀軒助教授課)</a> 

                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>4/4</span></td>
                                        <td><span></span>
                                            國定假日
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>4/11</span></td>
                                        <td><span></span>
                                            期中考週
                                        </td>
                                       <td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>4/18</span></td>
                                        <td><span></span>
                                            如何正確微調模型 
                                        </td>
                                        <td>
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/Z6b5-77EfGk">Post-training and Forgetting</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/post_training.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/post_training.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        	<!--<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/mpuRca2UZtI">如何在多張GPU上訓練大型模型（王秀軒助教授課)</a> -->

                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>4/25</span></td>
                                        <td><span></span>
                                            AI 推理能力哪裡來 
                                        </td>
                                        <td>
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/bJFtcwLSNxI">Reasoning</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/reasoning.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/reasoning.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>5/2</span></td>
                                        <td><span></span>
                                            模型編輯 
                                        </td>
                                        <td>
                                        	
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/s266BzGNKKc">reason_eval</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/reason_eval.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/reason_eval.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                            <br>
                                            
                                            <i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/ip3XnTpcxoA">reason_shorter</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/reason_shorter.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/reason_shorter.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                            <br>
                                            
                                            <i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/9HPsz7F0mJg">edit</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/edit.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/edit.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>5/9</span></td>
                                        <td><span></span>
                                            Tomofun 團隊演講 (業界經驗分享) 
                                        </td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>5/16</span></td>
                                        <td><span></span>
                                            模型融合 
                                        </td>
                                        <td>
                                        	
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href=" https://youtu.be/jFUwoCkdqAo">Model merging</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/merging.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/merging.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        	<span>NVIDIA 團隊演講 (RAG + Reasoning)</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>5/23</span></td>
                                        <td><span></span>
                                            生成策略
                                        </td>
                                        <td>
                                        	<i class="fa-brands fa-youtube"></i><a target="_blank" rel="noopener noreferrer" href="https://youtu.be/gkAyqoQkOSk">Speech</a> 
                                        	<i class="fa-regular fa-file-powerpoint"></i>
                                            <a href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/speech.pptx"
                                               rel="noopener noreferrer" target="_blank">ppt</a> /
                                            <i class="fa-regular fa-file-pdf"></i><a
                                                href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/speech.pdf"
                                                rel="noopener noreferrer" target="_blank"> pdf</a>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>5/30</span></td>
                                        <td><span></span>
                                            國定假日 
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                        <td>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>6/6</span></td>
                                        <td><span></span>
                                            期末考週
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
                                        <td><span>2/21</span></td>
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
                                        <td><span>李冠儀、許景淯</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>x</span></td>
                                        <td><span>2/21</span></td>
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
                                        <td><span>2/21</span></td>
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
                                        <td><span>吳典叡</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>Bonus</span></td>
                                        <td><span>2/21</span></td>
                                        <td><span>Bonus Competition</span>
                                        </td>
                                        <td><span>
                                            <a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://youtu.be/7-tzsWnGm-w">
                                            </a>
                                        </span>
                                        </td>
                                        <td><span>
                                            <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/bonus.pdf">
                                            </a>
                                            
                                        </span>
                                        </td>
                                        <td><span>N/A</a></span>
                                        </td>
                                        <td>
                                        	<span>
                                        	<a class="fa-brands fa-kaggle fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/t/bf7880c9aa95483580e0c7a500b122b2">
                                            </a>
                                        	<a class="fa-brands fa-kaggle fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/competitions/ml-2025-bonus-competition/overview">
                                            </a>
                                        	</span>
                                        </td>
                                        <td><span>06/13/2025 23:59</span>
                                        </td>
                                        <td><span>陳宥林</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 1</span></td>
                                        <td><span>2/21</span></td>
                                        <td><span>AI Agent1 -- RAG</span>
                                        </td>
                                        <td>
                                        	<a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://youtu.be/0ylc6rnoTOM">
                                            </a>
                                        </td>
                                        <td>
                                        <span>
                                            <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data//hw1.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td>
                                        <span><a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1OGEOSy-Acv-EwuRt3uYOvDM6wKBfSElD?usp=sharing"></a></span>
                                                
                                                <span><a class="fa-brands fa-kaggle fa-xl"
                                                     target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/u0ulin/ml2025-homework-1"></a></span>
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/homeworks/2/description">
                                            </a>
                                        </td>
                                        <td>
                                        	<span>03/21/2025 23:59</span>
                                        </td>
                                        <td>
                                        	<span>陳宥林、馮柏翰、劉建蘴</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 2</span></td>
                                        <td><span>3/7</span></td>
                                        <td><span>AI Agent2</span>
                                        </td>
                                        <td>
                                        	<a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://youtu.be/o4AT86nLcd0">
                                            </a>
                                        </td>
                                        <td>
                                        <span>
                                            <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data//hw2.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td>
                                        <span><a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1NmccKSzfQwf4m1Rzjb_kCXOup7hWbpsT?usp=sharing"></a></span>
                                                
                                                <span><a class="fa-brands fa-kaggle fa-xl"
                                                     target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/yuxianglin032/ml2025spring-hw2-public"></a></span>
                                        </td>
                                        <td>
                                        	<span><a class="fa-brands fa-kaggle fa-xl"
                                                     target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/t/486f3f5988bd4240b6476b08ab579d0a"></a></span>
                                        </td>
                                        <td>
                                        	<span>03/28/2025 23:59</span>
                                        </td>
                                        <td>
                                        	<span>林毓翔、李晨安、上官世昀</span>
                                        </td>
                                    </tr>
                                    <tr>
                                    </tr>
                                    <tr>
                                        <td><span>HW 3</span></td>
                                        <td><span>3/14</span></td>
                                        <td><span>Understand Transformer</span>
                                        </td>
                                        <td>
                                            <a class="fa-brands fa-youtube-square fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://youtu.be/etqpJe0SX70">
                                            </a>
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data//hw3.pdf">
                                            </a>
                                            
                                        </span>
                                        </td>
	                                        <td>
		                                        <span><a class="fa-brands fa-python fa-xl"
		                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1Ku_p27ml8QJ-Rd7FilZQvDA9axe68V7o?usp=sharing"></a></span>
		                                                
	                                        </td>
	                                        <td>
		                                        <a class="fa-solid fa-g fa-xl"
	                                                target="_blank" rel="noopener noreferrer" href="https://www.gradescope.com/login">
	                                            </a>
	                                      
	                                        </td>
	                                        <td>
	                                        	<span>04/04/2025 23:59</span>
	                                        </td>
	                                        <td>
	                                        	<span>傅啟恩、李冠儀、許景淯</span>
	                                        </td>
	                                	</td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 4</span></td>
                                        <td><span>3/21</span></td>
                                        <td><span>Training Transformer</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/fexqmEdAbAM">
                                            </a>
                                        	
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data//hw4.pdf">
                                            </a>
                                            
                                        </span>
                                        </td>
	                                        <td>
		                                        <span><a class="fa-brands fa-python fa-xl"
			                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/14OYqI7qJs0-4PTPZ87qnlmybtP8_5G3H?usp=sharing"></a></span>
		                                            
	                                        </td>
	                                        <td>
		                                        <a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/homeworks/5/description">
                                            	</a>
	                                        </td>
	                                        <td>
	                                        	<span>04/11/2025 23:59</span>
	                                        </td>
	                                        <td>
	                                        	<span>李晨安、林毓翔、林熙哲</span>
	                                        </td>
	                                	</td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 5</span></td>
                                        <td><span>3/28</span></td>
                                        <td><span>Fine tune is powerful</span>
                                        </td>
                                    
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/kB5tbSeosqE">
                                            </a>
                                        	
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data//hw5.pdf">
                                            </a>
                                            
                                        </span>
                                        </td>
                                        <td>
                                        <span><a class="fa-brands fa-python fa-xl"
			                                     target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1XTLX9o2QveOs71-njzw2ZLNcz7RIDhyX?usp=sharing"></a></span>
		                                            
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/home">
                                            	</a>
                                        </td>
                                        <td>
                                        	<span>04/18/2025 23:59</span>
                                        </td>
                                        <td>
                                        	<span>上官世昀、陳又華、鄭席鈞</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 6</span></td>
                                        <td><span>4/18</span></td>
                                        <td><span>Fine-tuning leads to Forgetting</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/60you8wXgbc">
                                           </a>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/b8fad34gpFY">
                                           </a> 
                                        	
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw6.pdf">
                                            </a>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw6_model.pdf">
                                            </a>
                                            
                                        </span>
                                        </td>
                                        <td>
                                		    <span><a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1sXopMDAT0nRrOTL52ECSPV07gKNoDn7n"></a></span>
                                                
                                            <span><a class="fa-brands fa-kaggle fa-xl"
                                                 target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/b10902031/ml2025hw6"></a></span>       
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/home">
                                            	</a>
                                        </td>
                                        <td>
                                        	<span>05/09/2025 23:59</span>
                                        </td>
                                        <td>
                                        	<span>馮柏翰、劉建蘴、吳典叡</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 7</span></td>
                                        <td><span>5/2</span></td>
                                        <td><span>RLHF</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/ZdqNmFW7ChM">
                                           </a>	
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw7.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td>
                                		    <span><a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1PHHQ6u4r7CwrbMPzDWY9lYw0Er0sbYcY"></a></span>
                                                
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-g fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://www.gradescope.com/courses/1000456">
                                            	</a>
                                        </td>
                                        <td>
                                        	<span>05/23/2025 23:59</span>
                                        </td>
                                        <td>
                                        	<span>鄭席鈞、袁紹翔、陳竣瑋</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 8</span></td>
                                        <td><span>5/9</span></td>
                                        <td><span>Model Editing</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/guyhx2QwLEQ">
                                           </a>	
                                        </td>
                                        <td>
                                        <span>
                                        <a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw8.pdf">
                                            </a>
                                        </span>
                                        </td>
                                        <td>
                                		    <span><a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1-HQkFyIHhZkYtRYWZO1RJCRss9DPGDX2?usp=sharing"></a></span>
                                                
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-g fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://www.gradescope.com/courses/1000456">
                                            	</a>
                                        </td>
                                        <td>
                                        	<span>05/30/2025 23:59</span>
                                        </td>
                                        <td>
                                        	<span>謝翔、上官世昀、陳又華</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 9</span></td>
                                        <td><span>5/16</span></td>
                                        <td><span>Model Merging</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://youtu.be/ZMUsSODHb7c?si=nqn_TY1ax263YF9Q">
                                           </a>	
                                        </td>
                                        <td>
                                        <span>
                                        	<a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw9.pdf">
                                            </a>
                                        </span>
                                        
                                        </td>
                                        <td>
                                		    <span>
                                		    	<a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/17CMSi43f2AS8Hc2Dr19hQlipMvFJu8xq?usp=sharing">
                                                </a>
                                            </span>
                                            
                                            <span>
                                            	<a class="fa-brands fa-kaggle fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://www.kaggle.com/code/monicahuangml1/ml2025-hw9-model-merging">
                                                </a>
                                            </span>
                                                
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/home">
                                            	</a>
                                        </td>
                                        <td>
                                        	<span>06/06/2025 23:59</span>
                                        </td>
                                        <td>
                                        	<span>黃筱穎、陳又華、謝翔</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><span>HW 10</span></td>
                                        <td><span>5/23</span></td>
                                        <td><span>Diffusion</span>
                                        </td>
                                        <td>
                                           <a class="fa-brands fa-youtube-square fa-xl"
                                           target="_blank" rel="noopener noreferrer" href="https://www.youtube.com/watch?v=raCxVQhtorI">
                                           </a>	
                                        </td>
                                        <td>
                                        <span>
                                        	<a class="fa-solid fa-file-pdf fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw10.pdf">
                                            </a>
                                        </span>
                                        
                                        </td>
                                        <td>
                                		    <span>
                                		    	<a class="fa-brands fa-python fa-xl"
                                                target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/drive/1V-gvtTbuqWCvIv7Ox4NE9gb_HiMEplVz?usp=sharing">
                                                </a>
                                            </span>
                                        </td>
                                        <td>
                                        	<a class="fa-solid fa-j fa-xl"
                                               target="_blank" rel="noopener noreferrer" href="https://ml.ee.ntu.edu.tw/home">
                                            	</a>
                                        </td>
                                        <td>
                                        	<span>06/13/2025 23:59</span>
                                        </td>
                                        <td>
                                        	<span>林熙哲、袁紹翔、劉建蘴</span>
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
                                    href="mailto:ntu-ml-2025-spring-ta@googlegroups.com">ntu-ml-2025-spring-ta@googlegroups.com</a>
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
