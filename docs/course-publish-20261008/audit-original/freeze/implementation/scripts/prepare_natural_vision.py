"""Prepare a pinned, attributed DOCCI scene-caption and question-answering teaching set.

This is an adaptation of Google DOCCI, not a synthetic-color task. Training uses
only official train images and the first complete human-authored English caption
sentence. Validation/test questions were manually written by an AI reader after
viewing every selected image and reading its full official caption, before any
model inference. These small held-out subsets cannot establish general VLM ability
or prove that a pretrained foundation model has never seen the images.

Every network download has an immutable repository revision or GCS generation,
plus a required SHA256. Running twice produces identical content. Dataset rows
only contain relative image paths. No paid service or model training is invoked.
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

from PIL import Image

DATASET_VERSION = "docci-natural-scenes-v3-144"
HF_REVISION = "a0a43eaf34676ffd008fb6565dd8c2ba00d09100"
WEBSITE_REVISION = "c190366529943195d90c141899abb2ec68dc345d"
DESCRIPTION_GENERATION = "1714384012999810"
DESCRIPTION_SHA256 = "c9df4819963883af35ddd2cf257949892fd8c6d88b33a012094352df60719800"

TRAIN_IDS = [
    "train_07545",
    "train_04792",
    "train_02278",
    "train_09205",
    "train_08965",
    "train_02741",
    "train_09502",
    "train_05131",
    "train_05452",
    "train_00019",
    "train_03306",
    "train_04602",
    "train_01509",
    "train_03470",
    "train_00722",
    "train_04665",
    "train_07331",
    "train_03402",
    "train_08593",
    "train_06612",
    "train_05086",
    "train_03255",
    "train_06230",
    "train_05664",
    "train_01225",
    "train_03167",
    "train_06144",
    "train_09385",
    "train_02485",
    "train_08438",
    "train_01278",
    "train_04411",
    "train_06988",
    "train_05014",
    "train_01894",
    "train_00812",
    "train_02347",
    "train_05079",
    "train_03897",
    "train_03747",
    "train_01258",
    "train_01200",
    "train_02544",
    "train_00695",
    "train_07323",
    "train_01941",
    "train_02289",
    "train_03496",
    "train_00231",
    "train_04197",
    "train_00570",
    "train_03959",
    "train_02939",
    "train_03125",
    "train_08293",
    "train_06054",
    "train_07121",
    "train_04566",
    "train_07320",
    "train_04613",
    "train_08265",
    "train_07726",
    "train_07481",
    "train_08935",
    "train_00265",
    "train_07552",
    "train_01697",
    "train_04300",
    "train_02902",
    "train_01241",
    "train_07472",
    "train_06293",
    "train_03283",
    "train_09090",
    "train_07884",
    "train_06526",
    "train_00400",
    "train_01483",
    "train_08563",
    "train_09480",
    "train_04112",
    "train_04275",
    "train_01767",
    "train_05762",
    "train_07136",
    "train_04313",
    "train_08613",
    "train_07613",
    "train_01627",
    "train_09533",
    "train_08732",
    "train_01757",
    "train_05725",
    "train_01749",
    "train_04035",
    "train_06277",
    "train_03212",
    "train_07906",
    "train_05487",
    "train_05671",
    "train_08709",
    "train_03287",
    "train_07518",
    "train_05606",
    "train_04439",
    "train_01704",
    "train_00890",
    "train_03630",
    "train_01091",
    "train_08875",
    "train_04976",
    "train_04695",
    "train_06468",
    "train_01316",
    "train_08163",
    "train_01935",
    "train_05561",
    "train_07130",
    "train_08671",
    "train_09368",
]

IMAGE_SPECS = [
    {
        "id": "train_07545",
        "cluster_id": "144",
        "generation": "1713028333269097",
        "sha256": "a5fdfff423c1372ed7711f8c49419fcc755d0c915ff158b6d192c39a75458ff9",
        "bytes": 131079,
        "dimensions": [1024, 768],
        "rgb_sha256": "a4ef0e23dfb25d266debf1b23e265383e5690f6cc903bb3a5d0c94f42e43e530",
    },
    {
        "id": "train_04792",
        "cluster_id": "41",
        "generation": "1713028331330361",
        "sha256": "2b11364ff3d2d82203572e195e22ba84e44de8130c11e57feb42c783caa147bd",
        "bytes": 164235,
        "dimensions": [1024, 768],
        "rgb_sha256": "bf60c2e86e1262a2f5d490b5af6d7488d6e30fd4e78122849324acd68b0e52cf",
    },
    {
        "id": "train_02278",
        "cluster_id": "8",
        "generation": "1713028329444252",
        "sha256": "684423d9b52e8563bd2e5fc648f21b3c116618c383c74489730c5b401072fced",
        "bytes": 164098,
        "dimensions": [1024, 768],
        "rgb_sha256": "b6f1a127abf0caefbf3f2bfe4acca8a5773d12e1cdd3c647746bcdacc60fd114",
    },
    {
        "id": "train_09205",
        "cluster_id": "27",
        "generation": "1713028334655161",
        "sha256": "a76b37f3c7c8de32be26233ff7327af31e61903346967cc43902fbae252bbeae",
        "bytes": 165903,
        "dimensions": [1024, 768],
        "rgb_sha256": "fd7506e6428c5182bdc39d7be22cb9050c91bb4de9e1e91af2fbe2bae1cfaf47",
    },
    {
        "id": "train_08965",
        "cluster_id": "39",
        "generation": "1713028334296770",
        "sha256": "7688e368ce89df6d58cb5dd2c97a8e1023598612a902fc3fed25d35d50c76880",
        "bytes": 132123,
        "dimensions": [768, 1024],
        "rgb_sha256": "482e540fdfc76f5eaa14fc5b7830dc5219ca8c13c26240d3b3d7c42f520753ae",
    },
    {
        "id": "train_02741",
        "cluster_id": "149",
        "generation": "1713028329774714",
        "sha256": "0f4b4d726db7f884b64371812f3d6d8acaa9d7a5a4eb70b679b95e74089ded15",
        "bytes": 111345,
        "dimensions": [1024, 768],
        "rgb_sha256": "c3ddeeab8eef2bc712afe090acc676c7dfc7eeab9a641f5092b3c9692cd38d32",
    },
    {
        "id": "train_09502",
        "cluster_id": "12",
        "generation": "1713028334828257",
        "sha256": "d79b0cefd5309b362796031c0a7b5c30d8622d7695c29af4690bd2735dba5910",
        "bytes": 107644,
        "dimensions": [1024, 768],
        "rgb_sha256": "52dfa871abd611f828cdbc595487d720c069e0ef37bf9953ac1d85af3b39a320",
    },
    {
        "id": "train_05131",
        "cluster_id": "4",
        "generation": "1713028331580777",
        "sha256": "63364ada9f588ede949940579d12dc222e06edc90b383b8fbb9001fa80b994e8",
        "bytes": 69471,
        "dimensions": [480, 360],
        "rgb_sha256": "ce34d6d21bf3f8230619e2a6b5308d81257b1c511d118d09ba7946fbb60ffecb",
    },
    {
        "id": "train_05452",
        "cluster_id": "31",
        "generation": "1713028331731145",
        "sha256": "0e8112d895b4cf83f786d9874c992047dcb127e8f1587fa8eff5ab59f89e6779",
        "bytes": 185195,
        "dimensions": [1024, 768],
        "rgb_sha256": "b2da18be85c91498d2465f928f0714912eb43af17243bdbab9d9687678bf7276",
    },
    {
        "id": "train_00019",
        "cluster_id": "137",
        "generation": "1713028328066128",
        "sha256": "2d0ef059707e88f73180a5da7c894abdd503ffc987d51dbab076886947035394",
        "bytes": 64328,
        "dimensions": [768, 1024],
        "rgb_sha256": "96cbe079f2462b80935c8fd12c0a324a94ccc544e349cdba9150100f11d135c1",
    },
    {
        "id": "train_03306",
        "cluster_id": "136",
        "generation": "1713028330182153",
        "sha256": "8a125ccc5b9f8a968878a88b49f5a91807cbe5f65b579aa5def5594d04fec323",
        "bytes": 132640,
        "dimensions": [768, 1024],
        "rgb_sha256": "e07ab42e6b6c00a3c20a64338ddc5ef3efb409e1a07258ce684b0dc47bdc8043",
    },
    {
        "id": "train_04602",
        "cluster_id": "95",
        "generation": "1713028331059153",
        "sha256": "2bf55e66ffa9cb48457ed8cb1affd1b790727df325ac19070c108f4455776c50",
        "bytes": 231659,
        "dimensions": [1024, 768],
        "rgb_sha256": "b9db34750861010ae9cdba8f4f23158da7b1eae87027682540b353e85da9a4c0",
    },
    {
        "id": "train_01509",
        "cluster_id": "38",
        "generation": "1713028329228912",
        "sha256": "a03018d434ea6a874fa8ad0840d173a6dabf2ad84d1355c0d333b0e23f45f927",
        "bytes": 48369,
        "dimensions": [480, 360],
        "rgb_sha256": "58883f7a11b0911eed1ad0497cf97934010a1aa46b9c0d0c85fd3956fab1a0b1",
    },
    {
        "id": "train_03470",
        "cluster_id": "110",
        "generation": "1713028330286296",
        "sha256": "989ae37cd4ae0345fb31ed648c3ae596231bd3169e822f4558d0ca9962f637cb",
        "bytes": 21804,
        "dimensions": [359, 480],
        "rgb_sha256": "a46bec5ab217decd78bfd8a0ed257de6be28f2be7d8522fff1e6236f8db44d09",
    },
    {
        "id": "train_00722",
        "cluster_id": "36",
        "generation": "1713028328505099",
        "sha256": "276c20f2e60c8489398eeca4641547c9096fbb1726a3d6a60aba8d451f5bf66b",
        "bytes": 106432,
        "dimensions": [768, 1024],
        "rgb_sha256": "a1e416fb303e948bc06b7b40903292ffa1a9e860b2e2304d19395937498e7a81",
    },
    {
        "id": "train_04665",
        "cluster_id": "23",
        "generation": "1713028331151844",
        "sha256": "e666675b492e2005940231c4daa9075368b13331f96994352308831bdda786bb",
        "bytes": 29964,
        "dimensions": [480, 360],
        "rgb_sha256": "c0a4cba05842c15c5e75b3b94a05eaf0b35c85fef54236787a7bd3a6ff61820a",
    },
    {
        "id": "train_07331",
        "cluster_id": "108",
        "generation": "1713028333170479",
        "sha256": "5d77d0d35fb22f3c026ec811609af50b4e7b784876e661fed430bccdcc09cc53",
        "bytes": 55219,
        "dimensions": [768, 1024],
        "rgb_sha256": "c632c6f70cdb440d21e5e8721b1602be76c08df59c808c6a509646bccba5de0c",
    },
    {
        "id": "train_03402",
        "cluster_id": "115",
        "generation": "1713028330200021",
        "sha256": "06703bcf0c0ce85ae84cf5d7a5668708fe2d51aeeec3f5af29957592b0ac16c6",
        "bytes": 100230,
        "dimensions": [1024, 768],
        "rgb_sha256": "f99b537f2f1ac224b5877b80c759adf2c6adcb29c8bbbd517b66fcf4b645d011",
    },
    {
        "id": "train_08593",
        "cluster_id": "57",
        "generation": "1713028333961486",
        "sha256": "2f30ba8a198b49b0c14f0327ca07a5f9d56af49b0ecb51922906843a05e7f3f3",
        "bytes": 109620,
        "dimensions": [1024, 768],
        "rgb_sha256": "5ff24269a3a01db0252969e5f4233f07434f5ca78f98e87435e84a6cab0371f8",
    },
    {
        "id": "train_06612",
        "cluster_id": "138",
        "generation": "1713028333973231",
        "sha256": "72434df7f5092f89d849e417d8d0294c4aad71584d9a4ff164fa5d2b44e6c2bb",
        "bytes": 59052,
        "dimensions": [480, 360],
        "rgb_sha256": "1f87858d32d4835e76b1efd00173a3d7c3a76ace90030929d3f2b6dc765d0c82",
    },
    {
        "id": "train_05086",
        "cluster_id": "9",
        "generation": "1713028331582837",
        "sha256": "21894f095a155a8c13d631355872e5c6d619ca4f56084a34a6dfc886967c9104",
        "bytes": 78126,
        "dimensions": [768, 1024],
        "rgb_sha256": "3885bc9eaecbe482bb265e2b8869ac4df58cddc4cb0f766ff7515f2b23629f3d",
    },
    {
        "id": "train_03255",
        "cluster_id": "126",
        "generation": "1713028330068799",
        "sha256": "ef5367325963f012b92ebc5543e0f83e69b6f08ef89af3e9348ed12fb7f795f8",
        "bytes": 201059,
        "dimensions": [1024, 768],
        "rgb_sha256": "00cf8483ec9388c14f2bea2b3161dd2444003e4ebee695e9e29aa203b593a51b",
    },
    {
        "id": "train_06230",
        "cluster_id": "111",
        "generation": "1713028332396437",
        "sha256": "bf024c3402dffb04cad7041dcd88c7651bb4d7c1b2e76babca241bd5b1e2b005",
        "bytes": 18628,
        "dimensions": [1024, 768],
        "rgb_sha256": "1f468555ca5078a77b97b51e960542fcbef8af304f95eff2a48f4b7b7eb13bde",
    },
    {
        "id": "train_05664",
        "cluster_id": "51",
        "generation": "1713028331875519",
        "sha256": "f6fcd987a02c71cf812a945a429cee3c13aeb20be2963e4e80b2384971dc7502",
        "bytes": 110489,
        "dimensions": [1024, 768],
        "rgb_sha256": "230797a9a9ed18076732ab278fb8309cde8012b1bea9bf70de955576a8db4fc8",
    },
    {
        "id": "train_01225",
        "cluster_id": "121",
        "generation": "1713028328910710",
        "sha256": "36b7e1381474e603684758c3fe16a9a8a6e5f462255eff22054efee4f91e0aa5",
        "bytes": 88190,
        "dimensions": [1024, 768],
        "rgb_sha256": "d19941b3a634a168bb376a7a00f1354c5e4b0b86dc51009797ae58f95e69e9a7",
    },
    {
        "id": "train_03167",
        "cluster_id": "64",
        "generation": "1713028330077334",
        "sha256": "cef02a74c12479ecd49c9c69d1c713fdb9e55d2535498f9523a162e0fb3deba8",
        "bytes": 126412,
        "dimensions": [1024, 768],
        "rgb_sha256": "dd94544b5197508ac309b329504b4c8e47fb94eb97e6504bcf0108fae916776b",
    },
    {
        "id": "train_06144",
        "cluster_id": "92",
        "generation": "1713028332225831",
        "sha256": "c172dbd3cc2a83f14a3689dff8891ae0d820342922f26dd19f3e174ac9ae0016",
        "bytes": 259072,
        "dimensions": [1024, 768],
        "rgb_sha256": "37ada85a6e218d5049e057e5a7d9704526620276ae343a6d4bf6559fc06c4dbb",
    },
    {
        "id": "train_09385",
        "cluster_id": "48",
        "generation": "1713028334602370",
        "sha256": "c00018672735236d5f3d96b9e3831e9de7e7f0f6bf02a861d5d48ef4ac093323",
        "bytes": 329989,
        "dimensions": [1024, 768],
        "rgb_sha256": "a4cf1e4c8930e583eea1b63372cdfdaea760bb9f71131aaf8e9d96bb4826eab3",
    },
    {
        "id": "train_02485",
        "cluster_id": "78",
        "generation": "1713028329679986",
        "sha256": "c94fda87a48400af42795bb7adf9c838a6edf24e418558bd65421efde012c0c6",
        "bytes": 197075,
        "dimensions": [1024, 768],
        "rgb_sha256": "f972ecf20237fb6917a8d5ce3c7ed8fb7b177d783466943a78eedd1849edb0f3",
    },
    {
        "id": "train_08438",
        "cluster_id": "7",
        "generation": "1713028333980834",
        "sha256": "e67bf9adbd98c5e9b8b4f2874aa7de427231f977383acb506b50b80c5a1e0496",
        "bytes": 178939,
        "dimensions": [768, 1024],
        "rgb_sha256": "f3944396ba66d292c8137208a529504c52fad1070a670ba758f940ab5b1d9bb9",
    },
    {
        "id": "train_01278",
        "cluster_id": "70",
        "generation": "1713028328969838",
        "sha256": "611921dee672e0d5625cfd7cc522bbce60174ec15e73d3f00c2ea50f2100bc0f",
        "bytes": 98568,
        "dimensions": [768, 1024],
        "rgb_sha256": "3d6aa9c23cedce9770b145e37b836c1ab46bb32ed941b26a109443f03c31405f",
    },
    {
        "id": "train_04411",
        "cluster_id": "120",
        "generation": "1713028331069515",
        "sha256": "63ffd14aed784433ac1e7e27f069bc2db8dca1d19504058faf0b80b3cf0a2887",
        "bytes": 125824,
        "dimensions": [1024, 768],
        "rgb_sha256": "5bed467647f8821e7798a722c7ec3be051c1eb95ac034471d62e01722eb8be40",
    },
    {
        "id": "train_06988",
        "cluster_id": "66",
        "generation": "1713028332850661",
        "sha256": "4039dd76b8a22fbda8ce5266441c951aed2f6644359a732f74713b9ece48a3c1",
        "bytes": 25179,
        "dimensions": [1024, 768],
        "rgb_sha256": "c3a1a2467cc1c0b09c3bdbfa1fec4988be94284046553d5db2ff63e81494f121",
    },
    {
        "id": "train_05014",
        "cluster_id": "59",
        "generation": "1713028331439467",
        "sha256": "8dc707284bb5f29706754c54908bad26de07c8aeb612330c09e21cbd4de5d2d3",
        "bytes": 139594,
        "dimensions": [1024, 768],
        "rgb_sha256": "7840eb76beb0aaaaaaa8bfe905e41525cc5368a1fe19196dfe89ea2798be888d",
    },
    {
        "id": "train_01894",
        "cluster_id": "77",
        "generation": "1713028329267952",
        "sha256": "77fbcf9060b24099be9199ebb2d4a591e9112f2ad2ce0b5adde59438ec76589a",
        "bytes": 280043,
        "dimensions": [768, 1024],
        "rgb_sha256": "e2b06a9e67b830c245b7bd8a9e889d7ac1bd564ec244dc89e586aae0a300e20b",
    },
    {
        "id": "train_00812",
        "cluster_id": "102",
        "generation": "1713028328338877",
        "sha256": "b4b7c5c80928e8eaa579e39febdfaa09969ceeef68efe50c84ad45210ac504ff",
        "bytes": 227872,
        "dimensions": [1024, 768],
        "rgb_sha256": "d68f0ccbd6382b1d1ba9a1fd4e72e728b7b5ba8797195eff8ebcac8b06906281",
    },
    {
        "id": "train_02347",
        "cluster_id": "85",
        "generation": "1713028329467677",
        "sha256": "92b48b8b3698b688ebdbfc0cd674f403cb9241859e6ade84dd9e59ff3aeeb3bb",
        "bytes": 94064,
        "dimensions": [1024, 768],
        "rgb_sha256": "ec0b845f50d9c0fd04a77a2dcd1bf0d62f9ac2cb9431c182a82895275f020140",
    },
    {
        "id": "train_05079",
        "cluster_id": "101",
        "generation": "1713028331662719",
        "sha256": "7e3cf114ccb233b7820b69a6949f9c1ef3e41f8c4cd2d9c5dab3af9a6177041f",
        "bytes": 105396,
        "dimensions": [1024, 768],
        "rgb_sha256": "9c5108b39605cbc13aec25ec84df4a2b03b8131e377b21600126f8d4f6b0f7e9",
    },
    {
        "id": "train_03897",
        "cluster_id": "125",
        "generation": "1713028330631119",
        "sha256": "054d233993a18e8dcd519e907a8787d157a066081b4a3914df3743778c4629ce",
        "bytes": 142210,
        "dimensions": [1024, 768],
        "rgb_sha256": "b17a945eda0fb8c8d2214ce2948103f03634dcb7c2d7930f97b0f436899d35f5",
    },
    {
        "id": "train_03747",
        "cluster_id": "37",
        "generation": "1713028330633348",
        "sha256": "6a390de7b4955b00000b5df350d1c49c9f21e7c92496f14c8a2bf5f0a5743eb3",
        "bytes": 45352,
        "dimensions": [768, 1024],
        "rgb_sha256": "de63702980e04e83945ee7983ce0687b8bfb4422cfe9200b3c6a84bedeefeed2",
    },
    {
        "id": "train_01258",
        "cluster_id": "58",
        "generation": "1713028328677695",
        "sha256": "380268d9c26ee9b2711e1e872938eb93fd341413260650f2e7f0125237bbd820",
        "bytes": 54106,
        "dimensions": [480, 359],
        "rgb_sha256": "0df49a9f04cc6df0f8be5303a3039e3783760b1d1d2aa274ffa9a6063c381487",
    },
    {
        "id": "train_01200",
        "cluster_id": "139",
        "generation": "1713028328710481",
        "sha256": "39a46986d160df7ceae4327599a7c0cb21f932ead4e6f9b8e3526ae943eeaf07",
        "bytes": 224287,
        "dimensions": [1024, 768],
        "rgb_sha256": "f349799837c60d1bd46b6f065245e85c16d0de8701551d36457590c8086d1943",
    },
    {
        "id": "train_02544",
        "cluster_id": "63",
        "generation": "1713028329552626",
        "sha256": "e15ff59e35d719e2feca13633ef61c11dc2c28686d72fcef127d44654dbac81a",
        "bytes": 66557,
        "dimensions": [1024, 768],
        "rgb_sha256": "79ec2a7238c61170d01d1bc57f3d1cb89ddf520301276d590564351a39cdc6f6",
    },
    {
        "id": "train_00695",
        "cluster_id": "132",
        "generation": "1713028328555218",
        "sha256": "104ab23efdbc22010a877497d9cb774d0cdbded1e8f0c51a75ff1695ffe14dd3",
        "bytes": 136893,
        "dimensions": [1024, 768],
        "rgb_sha256": "de83208fc09e613538bda77a23bd20920cf3ccbe0d90139b26af08d9126fdc2f",
    },
    {
        "id": "train_07323",
        "cluster_id": "131",
        "generation": "1713028333020387",
        "sha256": "08ebebd45abf94fb1ee8d47fec966a271a4a146e180b4311897d61bd0f7e68fa",
        "bytes": 180867,
        "dimensions": [1024, 768],
        "rgb_sha256": "5150c928e80f07ddf7330c195b5014ac0f7c0455590278d759b306e2e4b57322",
    },
    {
        "id": "train_01941",
        "cluster_id": "79",
        "generation": "1713028329496778",
        "sha256": "225601b69541c636a47de70f78ea3e7f10c2bd1170b44607bd4849afc0551ec4",
        "bytes": 140721,
        "dimensions": [1024, 768],
        "rgb_sha256": "98d3470106e662aaca643987112abe1343d683ca44a5587db67688b44b9173c8",
    },
    {
        "id": "train_02289",
        "cluster_id": "91",
        "generation": "1713028329564213",
        "sha256": "119447edd195ba6a66db02cd9dd0b0d0d6a9c2fa319f4e6a73a173a7f53d2568",
        "bytes": 67406,
        "dimensions": [1024, 768],
        "rgb_sha256": "166592f9a779ae0de3171151e3231b8a74635fcd1ef11035f036bae940ae018b",
    },
    {
        "id": "train_03496",
        "cluster_id": "69",
        "generation": "1713028330608694",
        "sha256": "5414d5d87280c6dca1bd5014d80ec267846a8a13362bca5f89ebfc85508f63cc",
        "bytes": 128073,
        "dimensions": [768, 1024],
        "rgb_sha256": "d8d84bd6f8b3e23b6ccc88161a3b89943291f4b718fc9031bcd2589c58f3c057",
    },
    {
        "id": "train_00231",
        "cluster_id": "3",
        "generation": "1713028328114775",
        "sha256": "2338a7e958e21602a1de6568badf4e9912baea41a578ce9883eb2b58652fbb29",
        "bytes": 163960,
        "dimensions": [1024, 768],
        "rgb_sha256": "8e451adb6b2e49a7befc9cb1226bf9d6c91f19c3f044e56ec7fa82368198fb13",
    },
    {
        "id": "train_04197",
        "cluster_id": "47",
        "generation": "1713028331218921",
        "sha256": "3914298c864556ddb2dd7afd88cfb4ebfeeed466e1e7442ada169b2b441b6f64",
        "bytes": 167415,
        "dimensions": [768, 1024],
        "rgb_sha256": "c5d96f87c4ed2154016cc52e802549c17d7d119466e23e9e5825b6ce6c8efb59",
    },
    {
        "id": "train_00570",
        "cluster_id": "62",
        "generation": "1713028328425952",
        "sha256": "384fbaae44e8046c0eae5acbc1834f26f0ed5893f489cb066b631e11d53da8b4",
        "bytes": 129394,
        "dimensions": [768, 1024],
        "rgb_sha256": "d9cebe2cc2d6e1d338b64d6fccdbb79182f4b012ae58ba776c93bfa7198851ac",
    },
    {
        "id": "train_03959",
        "cluster_id": "1",
        "generation": "1713028330738990",
        "sha256": "73c582e4d0b0d196ec215c729d9465379f89762e47c1474dd58c2a2891933c15",
        "bytes": 126120,
        "dimensions": [1024, 768],
        "rgb_sha256": "645879e86f80e577310b0ed920c7dd90fffebd54217a9a8220e8e76e02123645",
    },
    {
        "id": "train_02939",
        "cluster_id": "82",
        "generation": "1713028330248567",
        "sha256": "09475556dc2babbc77ecbb0c78013d690d26996e43dec342496b688e86d6f2be",
        "bytes": 238277,
        "dimensions": [1024, 768],
        "rgb_sha256": "a9de797e38d22772a0d2fd561a83b00eb5d3e8ac278be3520295dca92a455b71",
    },
    {
        "id": "train_03125",
        "cluster_id": "10",
        "generation": "1713028330039696",
        "sha256": "1a37892deb77172099bf09277876d78bb338a02f1c5b429805c6473b4147c3ca",
        "bytes": 218209,
        "dimensions": [768, 1024],
        "rgb_sha256": "61c24af3256ae836a42645fe24be6f3b37d9fba0ca9a15c70f94b8d5f206c6c4",
    },
    {
        "id": "train_08293",
        "cluster_id": "28",
        "generation": "1713028333756748",
        "sha256": "aa4512c99888c729c797bc323695805bcde1f08f7d6606ff3e16779e41ce3620",
        "bytes": 28659,
        "dimensions": [1024, 768],
        "rgb_sha256": "dc077f7ddfd60a71b4c9f6fc3ee2037fbca92a9ce299f94e399c2e69876eca79",
    },
    {
        "id": "train_06054",
        "cluster_id": "54",
        "generation": "1713028332335570",
        "sha256": "0ced39e927ac2afa8e7f163aed187bb7b95c83cde9b8a19e63f347be93421172",
        "bytes": 159362,
        "dimensions": [768, 1024],
        "rgb_sha256": "4f7544e11d1fd0b6425d89b01d4b8abdac2e42ac9a0368f74d4fdf92a4595413",
    },
    {
        "id": "train_07121",
        "cluster_id": "140",
        "generation": "1713028332928907",
        "sha256": "c47cb8244fe36db337d0cfefc77732c26c2b92160c4530692c99c24416efce04",
        "bytes": 132907,
        "dimensions": [1024, 768],
        "rgb_sha256": "633945d07b2f779d3254c6fa82bed48d61f0a9a38a46a987f7882e1427fb7f8d",
    },
    {
        "id": "train_04566",
        "cluster_id": "133",
        "generation": "1713028331069366",
        "sha256": "c086fb83846fa113d16ce2f9059424d003fab0f3c8ad2c1f85dc7c0dae011f1e",
        "bytes": 165752,
        "dimensions": [1024, 768],
        "rgb_sha256": "b0332c5cf14575516d1ce7c43fc60b12bc47ea88b72946e7761dd00d9b674bf7",
    },
    {
        "id": "train_07320",
        "cluster_id": "25",
        "generation": "1713028333183365",
        "sha256": "86ba2d7d94b3da5eb668661aca496799fc2da3667b185ef98c694c13d87b2edc",
        "bytes": 83419,
        "dimensions": [768, 1024],
        "rgb_sha256": "0dfa8e1f40884e78ba73841eebcaa0330676391cac7bc5e2fb312ff7d58b5ba0",
    },
    {
        "id": "train_04613",
        "cluster_id": "14",
        "generation": "1713028331279068",
        "sha256": "4c581425784d7162221f2b9e4162b91cec8f61ac74e673ac9052d9a5a22f7387",
        "bytes": 189735,
        "dimensions": [1024, 768],
        "rgb_sha256": "28626b2996f0c6ad45b1280ed187fb18ea28f4cdf983595c409b45563501cab8",
    },
    {
        "id": "train_08265",
        "cluster_id": "106",
        "generation": "1713028333801940",
        "sha256": "4d2073358c11458d3469e9b56a668063614eabb66da49bd3bf0a54a271525490",
        "bytes": 298494,
        "dimensions": [1024, 768],
        "rgb_sha256": "7c5803a98ce54ed771fcdd5055b8c676001dbec10dd550c0b9f695b7ff449841",
    },
    {
        "id": "train_07726",
        "cluster_id": "100",
        "generation": "1713028333371775",
        "sha256": "d1c4f5df19253f9e7dfe07b6b6fc041e5952b32561489a79f90e66fc55c082ad",
        "bytes": 115927,
        "dimensions": [1024, 768],
        "rgb_sha256": "3a1f0d3778cd9b405b117b6da15e77bc2e7607de95e8654877bd06922a286c7a",
    },
    {
        "id": "train_07481",
        "cluster_id": "122",
        "generation": "1713028333241587",
        "sha256": "94789dd07ef89e17a061fab43390ebe855d8c214b38ce854dfe721329de03c14",
        "bytes": 32138,
        "dimensions": [1024, 768],
        "rgb_sha256": "9d5336571c7b62f5bb590105698d91549a90aede40c28d20bb17a46aa08e636c",
    },
    {
        "id": "train_08935",
        "cluster_id": "26",
        "generation": "1713028334302765",
        "sha256": "c96d90e718ab24c7d736ce3e2797937aaf240f2876ae346ec2a0e69e9e79bf84",
        "bytes": 16532,
        "dimensions": [480, 360],
        "rgb_sha256": "a293bd4b5f8eae2e8fbe2bcaa0552f88d14d6eb070d01791d872a106fcf93ee6",
    },
    {
        "id": "train_00265",
        "cluster_id": "29",
        "generation": "1713028328134121",
        "sha256": "2858cbb88bdc150ddb28f0423e2e7e4b98326113130aa14150c2bdaad2e76821",
        "bytes": 110223,
        "dimensions": [1024, 768],
        "rgb_sha256": "ba353e7da930f7683d23dfbd1dff0f52b77d0074c374173a3426e0d52b02f2f2",
    },
    {
        "id": "train_07552",
        "cluster_id": "98",
        "generation": "1713028333361605",
        "sha256": "af9e96069f1ffa5da5b22b38cd6d4fc5f8649bcf40d4fe90dc8a8bfb5a067012",
        "bytes": 226975,
        "dimensions": [1024, 768],
        "rgb_sha256": "9a6376303fd6163da4b0d26ad4706612387b0a051305ab9977caaa909a48c4e2",
    },
    {
        "id": "train_01697",
        "cluster_id": "17",
        "generation": "1713028329025440",
        "sha256": "4c97714d37c0374afc3ea762550a1b3b7345204ecaeb6beedb567c414985ad02",
        "bytes": 260241,
        "dimensions": [1024, 768],
        "rgb_sha256": "d286881616e0e304ff398ad53047926778df5bf6fb4dd80e2e01227070476833",
    },
    {
        "id": "train_04300",
        "cluster_id": "24",
        "generation": "1713028330779501",
        "sha256": "659a95b6cfe8ac61cea2dbf4eaf1db430e565b14bfa436b9ff3604623e22fbd0",
        "bytes": 104889,
        "dimensions": [1024, 768],
        "rgb_sha256": "7e226aaa42708aadc02e2fffcbf7718449de2be639c1a935bae48d30179b591a",
    },
    {
        "id": "train_02902",
        "cluster_id": "146",
        "generation": "1713028329875858",
        "sha256": "69a02935c28df3c343640b06b9d5cfee07489c3beb33bdce1766dfeec54c3ef6",
        "bytes": 144539,
        "dimensions": [1024, 768],
        "rgb_sha256": "0469125f89213d680c0c35ad0b1ba9689248ebd03dd5f956743175fbd4d053ec",
    },
    {
        "id": "train_01241",
        "cluster_id": "119",
        "generation": "1713028328683748",
        "sha256": "58e82057b9a28fa1b55c0fbda49cb549e07fb2418f4a8c8b53a21931c276c426",
        "bytes": 190580,
        "dimensions": [1024, 768],
        "rgb_sha256": "f244c530f54997c0403d23ccb83b553e1d7dd1f57736be69a128f99cfe5d186f",
    },
    {
        "id": "train_07472",
        "cluster_id": "16",
        "generation": "1713028333152241",
        "sha256": "fa75bd4d65cebad02940249f69f1d4294fb225d7a8b636733cd12ce0f96f1de5",
        "bytes": 135018,
        "dimensions": [768, 1024],
        "rgb_sha256": "b19ca758bdc5e3ae11e43c9fac68b398389e99f0a139ffabe18e428c9e31404b",
    },
    {
        "id": "train_06293",
        "cluster_id": "109",
        "generation": "1713028333630293",
        "sha256": "d904aed8fda5b14897f79a5dfdcb905d292609bed4e63552690163ecf99ad29c",
        "bytes": 107505,
        "dimensions": [1024, 768],
        "rgb_sha256": "288c710b6c38c4220558c98b3190418eb669f797ccd3a432799b88291fa24abd",
    },
    {
        "id": "train_03283",
        "cluster_id": "35",
        "generation": "1713028330095234",
        "sha256": "1599386b6f77900ca6f39fb3edbb091f976578e4dc43cd1d1aa824a5cb0e74aa",
        "bytes": 116301,
        "dimensions": [1024, 768],
        "rgb_sha256": "be29b58080090a6f88fb67244e2888002e2bfbd6c3a95ccfb753b65a036d5548",
    },
    {
        "id": "train_09090",
        "cluster_id": "46",
        "generation": "1713028334349594",
        "sha256": "a1641787c8108fa26cf740acd77da86ab5786e4791e3d9ad8f3782b49155a5bc",
        "bytes": 89059,
        "dimensions": [768, 1024],
        "rgb_sha256": "3c4d71bd3766437adb2528eed373dc01b0c422c9591c898e6f774d6f4d7e730e",
    },
    {
        "id": "train_07884",
        "cluster_id": "97",
        "generation": "1713028334690356",
        "sha256": "0cd4e68ef5760339ce6d4965c4184d0c26eebaf485f7430a4edb7c719f12d213",
        "bytes": 193329,
        "dimensions": [1024, 768],
        "rgb_sha256": "02b309d038840f8ea0288bc0d40586fd447259d753cca25e57cc86435ed71630",
    },
    {
        "id": "train_06526",
        "cluster_id": "128",
        "generation": "1713028332718457",
        "sha256": "93fc95e02660cefd366776f47063d641ed0eb323afb0f63021e629c980d9953f",
        "bytes": 264657,
        "dimensions": [768, 1024],
        "rgb_sha256": "affc3dd4052b6427c082f97d419add26c370195e5011bbc201c49f022ef2f2d9",
    },
    {
        "id": "train_00400",
        "cluster_id": "43",
        "generation": "1713028328241123",
        "sha256": "1b4d6602961400b6b7917a0d6cd8f0b49fd83a778ed56ba1c3ccf6d14ce13a41",
        "bytes": 72123,
        "dimensions": [1024, 768],
        "rgb_sha256": "265ef2bc1f9f4148915e7dbb88fe765efba87e806ec0b68cc6f48f88c0a3148c",
    },
    {
        "id": "train_01483",
        "cluster_id": "60",
        "generation": "1713028329175089",
        "sha256": "0a34f552ac57a9d95d68e017f48f57049cb024ef7ad2ce39db89fca6b1d8524b",
        "bytes": 122609,
        "dimensions": [1024, 768],
        "rgb_sha256": "e70ad935ae4ad9465a21b16758958f2fd9c20f989494d1c84181d8fa2c59ae5f",
    },
    {
        "id": "train_08563",
        "cluster_id": "56",
        "generation": "1713028334756469",
        "sha256": "0fb2f05e3270cf4c3ff1ab7da8e68f50c880c44d57e228b7df3f9f181e4b3d1b",
        "bytes": 233550,
        "dimensions": [768, 1024],
        "rgb_sha256": "3b4649e48565cfda1a0a480c43dd5051569e9fe2f5a153c324b76a68b858aae0",
    },
    {
        "id": "train_09480",
        "cluster_id": "99",
        "generation": "1713028334635432",
        "sha256": "ae62eafe1e7edd901218abd14eaeb8511d271810cf3e1fff92fa972a96d29427",
        "bytes": 106428,
        "dimensions": [1024, 768],
        "rgb_sha256": "e00d3dda3bf2541e1fbf49749d1c9c2992342d367fa442e18eeeab0f7ab81b36",
    },
    {
        "id": "train_04112",
        "cluster_id": "53",
        "generation": "1713028330724116",
        "sha256": "e2da5e00c9acc8aa92d7b3c0d7d180e74e5a6179b7e62c7d3adbaff236965a07",
        "bytes": 102393,
        "dimensions": [1024, 768],
        "rgb_sha256": "108e9a30049c469ce0f77bdb2f61ceb0c84b9c5ee3ad9f8d63240ef125a786be",
    },
    {
        "id": "train_04275",
        "cluster_id": "19",
        "generation": "1713028330868667",
        "sha256": "156a12a14679e59cfa63a287381b0ed70a0a3069aabce8960a742966d76b921d",
        "bytes": 251549,
        "dimensions": [1024, 768],
        "rgb_sha256": "df134420dbcf570140fc49dedf9cc7faee1c16593ae6e35132c0c43c2008d0dd",
    },
    {
        "id": "train_01767",
        "cluster_id": "143",
        "generation": "1713028329151937",
        "sha256": "1d365a5918566e7ae6c4ff409cea302ea8deabf9b5e64a7e0b529b8f2dce76ec",
        "bytes": 88274,
        "dimensions": [1024, 768],
        "rgb_sha256": "49db0487c661bbbe76a3d4f061c24173ad2e4446516001c7d8067524f5f5e7bc",
    },
    {
        "id": "train_05762",
        "cluster_id": "22",
        "generation": "1713028332076805",
        "sha256": "9e792be4a3b34bfa8f3080bf69456dbe9ca14473dcaab5d55a72f8e426d1f61a",
        "bytes": 166808,
        "dimensions": [1024, 768],
        "rgb_sha256": "eed36838522eb20cb5c5e6b8645740e5b5c1593db536354653f399ec821e2744",
    },
    {
        "id": "train_07136",
        "cluster_id": "96",
        "generation": "1713028332917924",
        "sha256": "f8d861275f2f24b344793efbacc2a0f5534462c699560a7661f29b4379bbc510",
        "bytes": 236943,
        "dimensions": [1024, 768],
        "rgb_sha256": "97c04f095efa14280fe52c4a2a1b67e6d92a66e81a98e3d76d5d83615aad0784",
    },
    {
        "id": "train_04313",
        "cluster_id": "13",
        "generation": "1713028331275868",
        "sha256": "018a4e108c73da6a076d6b19ebb05236277198849286103c9240f2cdd3c16398",
        "bytes": 118001,
        "dimensions": [768, 1024],
        "rgb_sha256": "d8fe20faa1856f62cc497cd5a6d951d8dc0999d2b98c896257bc7c7844421f59",
    },
    {
        "id": "train_08613",
        "cluster_id": "147",
        "generation": "1713028334038009",
        "sha256": "916b24a8bc2f0e9d896e0628d810e6e540ce06321bc0a664cafa982f81ee95dc",
        "bytes": 123235,
        "dimensions": [1024, 768],
        "rgb_sha256": "1eab49ce6123e0e1c7fe5d585687ac170e554dfb8f831de8168a91bd77192cce",
    },
    {
        "id": "train_07613",
        "cluster_id": "112",
        "generation": "1713028333289673",
        "sha256": "6f9e72ad82ab0279e211fb80d2239b96c9a3dd27c25d08c8d10357cd660d8a1d",
        "bytes": 32167,
        "dimensions": [1024, 768],
        "rgb_sha256": "a8b2cdd77938974f96c382158d44ce33d96ead6227260cdaeb1d13a12531d742",
    },
    {
        "id": "train_01627",
        "cluster_id": "130",
        "generation": "1713028328950025",
        "sha256": "ab0cf96b27b188f02dd9b2bccd827752b7e3997ed9ce2118b88d6bb24ceb413f",
        "bytes": 141485,
        "dimensions": [1024, 768],
        "rgb_sha256": "83ad1bbb3174c3ac70d52ebe7ef1200977c82ba6fa9d777c46398d6c58b4dc3d",
    },
    {
        "id": "train_09533",
        "cluster_id": "145",
        "generation": "1713028334726538",
        "sha256": "7832cfebce361bb6c358ed962c43943eb97bce3249d8171030380e0616fb28f2",
        "bytes": 62775,
        "dimensions": [768, 1024],
        "rgb_sha256": "ffd65872d0251302cf993f980e2a4192a1de7c8dd1c824adbc66c68cb69037bc",
    },
    {
        "id": "train_08732",
        "cluster_id": "34",
        "generation": "1713028334091816",
        "sha256": "cbde221d4ea226d8f34acdc59af139bf2af08641b422ef8002db47241d07b929",
        "bytes": 157999,
        "dimensions": [767, 1024],
        "rgb_sha256": "d715c037252cbe60ff53ab381b8abefc234ff688375539ec63cc8f843dc6363a",
    },
    {
        "id": "train_01757",
        "cluster_id": "21",
        "generation": "1713028329086494",
        "sha256": "a777fff0d83e07bb7d17fd531f8209b8bf180f1d2a3b345e51705a21e37b6166",
        "bytes": 148551,
        "dimensions": [1024, 768],
        "rgb_sha256": "861e246ef88d467245626bad95bca425e9eb565ea5b9f0e51516b7d79a049d3a",
    },
    {
        "id": "train_05725",
        "cluster_id": "40",
        "generation": "1713028331979829",
        "sha256": "3f8f5eb2e38f5efffe022f5200d779d732196ad6c42fd8bf9ff35889735afdd7",
        "bytes": 220594,
        "dimensions": [1024, 768],
        "rgb_sha256": "5996ae67f3a4d45a89c633c3d9f1c4ffd25c3f18f7bded571acd00dfa912468f",
    },
    {
        "id": "train_01749",
        "cluster_id": "116",
        "generation": "1713028329020803",
        "sha256": "e7f99104c7e5fb7e6a68d1247cdfcfc69924c663ab3975d236d7b22b45d7c1a9",
        "bytes": 118130,
        "dimensions": [1024, 768],
        "rgb_sha256": "b2ff7af11214bff8a4d941d0aa5f981f387f7dbae9c8b933b9d1e432c6355acc",
    },
    {
        "id": "train_04035",
        "cluster_id": "107",
        "generation": "1713028331011712",
        "sha256": "8845ef7b3252375b33998b95daf3117e4cd3b6038555fd7aea501876d6356119",
        "bytes": 131896,
        "dimensions": [768, 1024],
        "rgb_sha256": "a8476bbe3062cd83548e284fd6041f51ed04c4bef43e845a06131dcee3c172ce",
    },
    {
        "id": "train_06277",
        "cluster_id": "50",
        "generation": "1713028332419262",
        "sha256": "5992e857d73a620f3f65b6bff778eebf53e3d5985b4c352e5bda7df147041400",
        "bytes": 241604,
        "dimensions": [1024, 768],
        "rgb_sha256": "5b1eb418fdd47a3a1dec71517dd8a426ab370f1e952bc77bdf903d0083984f54",
    },
    {
        "id": "train_03212",
        "cluster_id": "87",
        "generation": "1713028330158512",
        "sha256": "c52fa78700412c14de3a6325ec0679bd863b8bfb34a04b583f85d6f05b62a680",
        "bytes": 109373,
        "dimensions": [768, 1024],
        "rgb_sha256": "751ed78fb8ec25a553d2367f6dc6dbdda5536fb14693c17bc046dd6646f26edc",
    },
    {
        "id": "train_07906",
        "cluster_id": "44",
        "generation": "1713028333553556",
        "sha256": "4825bc203951b0921e1da543d1ce61cb33f879993cbca90fa02f98d8c80e8022",
        "bytes": 180962,
        "dimensions": [767, 1024],
        "rgb_sha256": "81358325419d2f6bf936acaa63bad3e9ebefe69c8db9f321e6eff5915f84b6cf",
    },
    {
        "id": "train_05487",
        "cluster_id": "6",
        "generation": "1713028331908058",
        "sha256": "5c04b1a402988fb418f381daed71fc16f1671624c032d4f0ebd72eda4d544d0d",
        "bytes": 230571,
        "dimensions": [768, 1024],
        "rgb_sha256": "98747145e87ee3fe1a17e1bad3f14c70fd6efced4f37650fbbee35e6dde62d5c",
    },
    {
        "id": "train_05671",
        "cluster_id": "114",
        "generation": "1713028332046955",
        "sha256": "94d5174333d5b7917bc689c07768b4c27343e303f642f9de36bcb7e049e38f15",
        "bytes": 204114,
        "dimensions": [1024, 768],
        "rgb_sha256": "7839f986f2fc2689d3c394c054067e0941f8ecf672e253144c3300cd7c87b9ff",
    },
    {
        "id": "train_08709",
        "cluster_id": "73",
        "generation": "1713028334078327",
        "sha256": "782c7f2d952eac91ca4c1f379f35871a8e236eb4c601cbdc194584e03c580dac",
        "bytes": 178910,
        "dimensions": [1024, 768],
        "rgb_sha256": "b56c19364dbce114cc524671386a7990ed565b40e95a5c939cc310a2144d96b9",
    },
    {
        "id": "train_03287",
        "cluster_id": "5",
        "generation": "1713028330155355",
        "sha256": "f2d4680e8049f977631c6911718e0ea760ddd89107b283f3c3d1c676e30aa4ad",
        "bytes": 152714,
        "dimensions": [768, 1024],
        "rgb_sha256": "5d924c46fef9db9920371dcb4818e2156f14daf3979102ffa7e11249ea2bbd66",
    },
    {
        "id": "train_07518",
        "cluster_id": "74",
        "generation": "1713028333273456",
        "sha256": "980549ca2ccae9fc4870747ea487ad6e7cb7dda3220f08f5044dbb4702c4a999",
        "bytes": 165612,
        "dimensions": [768, 1024],
        "rgb_sha256": "550a678654ac0d9209b3f70bf50c60a577368d555981df9fed3937fb1abceeff",
    },
    {
        "id": "train_05606",
        "cluster_id": "67",
        "generation": "1713028331916419",
        "sha256": "84614a217b8462390c673c4d34e599c2234a4f3bc7177bc7c0c3d26829c3ae93",
        "bytes": 108192,
        "dimensions": [768, 1024],
        "rgb_sha256": "6c7e88074bf4e781f4b016cb7bb4e7f9e7cddf1f15a921fcf6a2655469fb4de2",
    },
    {
        "id": "train_04439",
        "cluster_id": "71",
        "generation": "1713028330968448",
        "sha256": "4728d5407e7124df1ef01252d0615218cff0acad131218c34ab233ec95a4da87",
        "bytes": 261114,
        "dimensions": [1024, 768],
        "rgb_sha256": "430ce77ad0db47cc9b20f7872b0162e02cbaa08f35ef3da18710e257dab19464",
    },
    {
        "id": "train_01704",
        "cluster_id": "127",
        "generation": "1713028329136367",
        "sha256": "a99fc1656bcc08913d7e4b10b6b965b35ef5cd9bde9574a65dca1e9960c7d0a1",
        "bytes": 155033,
        "dimensions": [1024, 768],
        "rgb_sha256": "81056cf4a1bc48f1350742d1544418288da9841b3c3cac6bb147d5bc2aa53734",
    },
    {
        "id": "train_00890",
        "cluster_id": "86",
        "generation": "1713028328605103",
        "sha256": "b7d9d6583e1dcbc6b0c221bcdc81a51307832ce74975d5a720722a2a9da7c35f",
        "bytes": 134212,
        "dimensions": [1024, 768],
        "rgb_sha256": "1410a1761236154f7dd4c9300ed821d58917b3f5deff608cfe16a4b09597ad5b",
    },
    {
        "id": "train_03630",
        "cluster_id": "88",
        "generation": "1713028330487115",
        "sha256": "f0a8dd1f90274a4f74359267258c5a46d04dcd801c43dfe58ffd107dae459c31",
        "bytes": 131664,
        "dimensions": [1024, 768],
        "rgb_sha256": "a3dac1297c9ac140b31ad9e74cb626ccfb96996bc379ce9986f5295e28f62300",
    },
    {
        "id": "train_01091",
        "cluster_id": "18",
        "generation": "1713028328564585",
        "sha256": "ba31ccf8f9d574a2933ee62d00c95f4bb92a86e499f75b6d425d0146eddfc874",
        "bytes": 281776,
        "dimensions": [1024, 768],
        "rgb_sha256": "f64172e724e809be493c3692d741df43423341f33b07e833b0cd49abf2c013e5",
    },
    {
        "id": "train_08875",
        "cluster_id": "93",
        "generation": "1713028334218740",
        "sha256": "5751d53f7e34bc0393e1c1c179049d47bb18c1a4f3ceda5d6ebc0ae99e9c83b8",
        "bytes": 36074,
        "dimensions": [360, 480],
        "rgb_sha256": "b1f5da36988740892e16dd5adf73721b16ad2a4364354c6a58aa789ad92f3d83",
    },
    {
        "id": "train_04976",
        "cluster_id": "104",
        "generation": "1713028331536571",
        "sha256": "3adf8e92a6586d670840d59092ff2ced35efa33ee1fd373345c57d1b2a840852",
        "bytes": 317353,
        "dimensions": [768, 1024],
        "rgb_sha256": "77f8df175efa81b2a4f9497f10a39686b7521bd5a6269ff9eb1e81fcae09d380",
    },
    {
        "id": "train_04695",
        "cluster_id": "32",
        "generation": "1713028331180948",
        "sha256": "9d7286f884e1baf7635cc4a35ddaaddf333038bbe21890e40fa8b9a527ae7317",
        "bytes": 54414,
        "dimensions": [767, 1024],
        "rgb_sha256": "7a55f3edb3d21e314aa7674d8e8828bec94d90d444c81b4095eb0b4e2de3fefb",
    },
    {
        "id": "train_06468",
        "cluster_id": "94",
        "generation": "1713028332481593",
        "sha256": "ac1f4386c8ca548a1026210e8c915ab83c23759b122562328c9e3192b8924af6",
        "bytes": 70765,
        "dimensions": [1024, 768],
        "rgb_sha256": "9f38339faac0e8f1b756eb9d18187ca8ae7e0e5b7c6b6e7d949cdeeb33eff583",
    },
    {
        "id": "train_01316",
        "cluster_id": "142",
        "generation": "1713028328710465",
        "sha256": "4b05be4bbcb81a39deb95df43a253041db1c01e3b65da607972ef0f7b8bb86bb",
        "bytes": 30857,
        "dimensions": [480, 358],
        "rgb_sha256": "970406fc12de319811895afb41b464e1dc7bbe6dbd23878bbcafe20562a2b053",
    },
    {
        "id": "train_08163",
        "cluster_id": "124",
        "generation": "1713028333688502",
        "sha256": "5cfd2d7d3b9807a60a09f828e726bc8102d5df41b35189100876da7943704419",
        "bytes": 127209,
        "dimensions": [1024, 768],
        "rgb_sha256": "f0f95d6fab5913faa76151264923024fa47f48d4198e6dcacf7a45c792fd257e",
    },
    {
        "id": "train_01935",
        "cluster_id": "81",
        "generation": "1713028329264655",
        "sha256": "18b0fe65f088c87a1a67e3507c0267f2d783c2da5e84f0fa710732851b084284",
        "bytes": 204404,
        "dimensions": [1024, 768],
        "rgb_sha256": "de67c8769a897417d0be92e90db1df69e8c9111052915342d29b27d0e28021a8",
    },
    {
        "id": "train_05561",
        "cluster_id": "61",
        "generation": "1713028331948430",
        "sha256": "a7806be267e1ebc7d2accb8acff03d5cc33a57167194aa725006b3beb1e78e38",
        "bytes": 103676,
        "dimensions": [768, 1024],
        "rgb_sha256": "2f0ca5847c190ffa38978a675ff3492ffddd05f0f519c13bb7d539b180f058b3",
    },
    {
        "id": "train_07130",
        "cluster_id": "148",
        "generation": "1713028332911811",
        "sha256": "0f2f5b4db7b9b039b2bc8a5bac72bcc819f08f97233d56bb8f9a7d2f0a29addc",
        "bytes": 140561,
        "dimensions": [1024, 768],
        "rgb_sha256": "3a22ca088a15dd4033feaab6a432fc7fa4f80b549c33764506fcde63edd7b44c",
    },
    {
        "id": "train_08671",
        "cluster_id": "118",
        "generation": "1713028334760942",
        "sha256": "21e602ad34f812ce4518e73da75e229af3fb8257df3c264be96e8bf14506085f",
        "bytes": 292579,
        "dimensions": [1024, 768],
        "rgb_sha256": "5fd77785870617fe7c4fc15309fed67ef20b00c04e611f0b9401038f143c1833",
    },
    {
        "id": "train_09368",
        "cluster_id": "72",
        "generation": "1713028334595975",
        "sha256": "38ea31a4c65ef77da399580e20ec384d89e31dfdf651e0330194692309b41f13",
        "bytes": 169886,
        "dimensions": [768, 1024],
        "rgb_sha256": "17be22346beaef779ec13e78b6c2b2ff5732951719f8f5145992d7dd2c132da4",
    },
    {
        "id": "test_00729",
        "cluster_id": "31",
        "generation": "1713028322603329",
        "sha256": "8fff2387a62580e6716d9e312ffd32e5058ae07af051d466c7aff941f6a94b60",
        "bytes": 112881,
        "dimensions": [1024, 768],
        "rgb_sha256": "f985d61d5c37d8290b33f5f2e764e21463e27bc0abb968b9552c87390461f347",
    },
    {
        "id": "test_01103",
        "cluster_id": "140",
        "generation": "1713028324564551",
        "sha256": "6db6e5306f9ee2f46e86667cc87705a59c40826b45773a3d3397ed60ca3ee9f7",
        "bytes": 107589,
        "dimensions": [767, 1024],
        "rgb_sha256": "884845ecfb7759e5baea61980680bdb9c319b5f015e38345b597d524e7ba6243",
    },
    {
        "id": "test_00706",
        "cluster_id": "80",
        "generation": "1713028322322450",
        "sha256": "5504201ea2453ae3b7c01e10dceba996c0fc93056e3af0909f54577857a8285e",
        "bytes": 84816,
        "dimensions": [1024, 768],
        "rgb_sha256": "f64cce5e19dc48ee82c6a7013ee0c4af9905b692d3ab046d6ee2daec42e2a9e5",
    },
    {
        "id": "test_00337",
        "cluster_id": "134",
        "generation": "1713028322013642",
        "sha256": "84d3815b597b20efc2fcb626dbe2d7e522fdfc73a378490714beeffd77e12eb4",
        "bytes": 88025,
        "dimensions": [1024, 768],
        "rgb_sha256": "bc14784bf8ae650cd170e1e564ec9c2b4a8c3c7d04f8ecd14210bcf696807f85",
    },
    {
        "id": "test_00091",
        "cluster_id": "50",
        "generation": "1713028322789974",
        "sha256": "f60dfc30fa680a63f9f8cc266bad05d4345bf19a82ac44573da291f7c2ec7318",
        "bytes": 95135,
        "dimensions": [1024, 768],
        "rgb_sha256": "a94370c17a71b58a602b698bd3f1af997bf87db7d9ae94b255bfa3f8bc0890ff",
    },
    {
        "id": "test_00039",
        "cluster_id": "2",
        "generation": "1713028322762560",
        "sha256": "9dc4d4a711f1233d9b82834768f0a186858e7b89238bc42e2002d2ba20a879e8",
        "bytes": 159353,
        "dimensions": [1024, 768],
        "rgb_sha256": "eab8fddc3b129c653d70ca369fca863b053e10f4cae9ffa839ee78ad2c68b858",
    },
    {
        "id": "test_00010",
        "cluster_id": "99",
        "generation": "1713028322368437",
        "sha256": "b7a40057d00e95a6811c3579e1608349160a2b409fe168d0758f7e2de776af40",
        "bytes": 170352,
        "dimensions": [1024, 768],
        "rgb_sha256": "c8fb6094cace2f1f7511bcec733e34c3f07f4133947bc1882c43852a4ef0ae90",
    },
    {
        "id": "test_00006",
        "cluster_id": "47",
        "generation": "1713028322038459",
        "sha256": "5448ecae5857d3acdb8d0c1bde88a375349fe48f1292011a1fc912cc93a3308f",
        "bytes": 146886,
        "dimensions": [1024, 768],
        "rgb_sha256": "0d0d6b67001656e84a5d0a7fbc23d2383bf9ee4f9aed8070822d72fa5a85db8c",
    },
    {
        "id": "test_00002",
        "cluster_id": "36",
        "generation": "1713028322671913",
        "sha256": "d7bf9e01622d30a24eeb8d4357922efb2b5ce853c40ae8e7af808e5b53553a59",
        "bytes": 75027,
        "dimensions": [1024, 768],
        "rgb_sha256": "9162499d1a7eba365094b488524898a18bad35bf011ab53983780f55deb95ddf",
    },
    {
        "id": "test_01629",
        "cluster_id": "14",
        "generation": "1713028324857048",
        "sha256": "a3af8387901821b8d1b24ce3325abfdb7a28695de65617a4bce55f52b16d99b7",
        "bytes": 196587,
        "dimensions": [768, 1024],
        "rgb_sha256": "ae54907f513ef4f98bc599f9488c390b2e1f8752f41f895581f4031fccdabc5d",
    },
    {
        "id": "test_00094",
        "cluster_id": "50",
        "generation": "1713028322826358",
        "sha256": "d842f54ce4bd60c5abeafe3b02bccac79f01441b0ee11708fdaaba1e57468f9b",
        "bytes": 37146,
        "dimensions": [768, 1024],
        "rgb_sha256": "1829fe839269a0c608af6ff54b330953d3466587bbf25796717ded1edef9023d",
    },
    {
        "id": "test_00048",
        "cluster_id": "28",
        "generation": "1713028321835459",
        "sha256": "abb535cacd3701bdfa7ba852d71ba5665f64e32fa680e945ca6c92400d3b4feb",
        "bytes": 46900,
        "dimensions": [768, 1024],
        "rgb_sha256": "529b1e4ee43a65793baaa5f25b98b84444b41dd4d4dcaaf95faf0521dc78329d",
    },
    {
        "id": "qual_dev_00000",
        "cluster_id": "130",
        "generation": "1713028322886058",
        "sha256": "f8a67385c1cbb17dc61ef60b6b22afe00822e91ba3a40be329a94b79f96aeecd",
        "bytes": 128244,
        "dimensions": [768, 1024],
        "rgb_sha256": "b0c8cd31121c4ba3a262fcf68bf902e378e203d7e2504eae6b31d7ea98c561a3",
    },
    {
        "id": "qual_dev_00003",
        "cluster_id": "51",
        "generation": "1713028322043923",
        "sha256": "addb796c66e600df5594ee220702c32c538730128c228abb63c789bfc6b8aec8",
        "bytes": 104799,
        "dimensions": [1024, 768],
        "rgb_sha256": "2de1b9d54bd88682802655d659d27ea519a4329ce7ac1a0a1b87132f7a887b9b",
    },
    {
        "id": "qual_dev_00014",
        "cluster_id": "83",
        "generation": "1713028322909525",
        "sha256": "d525e49335af5a418f12500d5f3cfa219bc02db40f26bff3edd83796393b93c9",
        "bytes": 173893,
        "dimensions": [1024, 768],
        "rgb_sha256": "d3688666d2038698fe93a9ce77b04270aa81645fd743a9ade9181b0c4eff0bfc",
    },
    {
        "id": "qual_dev_00015",
        "cluster_id": "29",
        "generation": "1713028322915175",
        "sha256": "f71bb1a65c60506e5bd57e363cb701351cb5839b71b5534362e5b3654e0462ca",
        "bytes": 123529,
        "dimensions": [1024, 768],
        "rgb_sha256": "d6becd5e8b253b564a2249d492a04f9e92e2b6790c45bc322f22d2f93f13e033",
    },
    {
        "id": "qual_dev_00017",
        "cluster_id": "43",
        "generation": "1713028322386004",
        "sha256": "a1b3d3296f34a389778a51b46302d2453bc93cd587e2b145ee777f6dda58c5b9",
        "bytes": 94174,
        "dimensions": [1024, 768],
        "rgb_sha256": "f5b787a41a51306fd4a275b0c9d957611ebfd4098117d733881c01e0231ebf6a",
    },
    {
        "id": "qual_dev_00021",
        "cluster_id": "140",
        "generation": "1713028322555144",
        "sha256": "97d3abd5f155ced3e281893f975dcf610849ffdcc6e91d9020b24e94b4d0cf15",
        "bytes": 118504,
        "dimensions": [768, 1024],
        "rgb_sha256": "adb72ba9b29bf14920d089df05c925dcc5f474c9dfb6a550206527f5b4cad6c7",
    },
    {
        "id": "qual_dev_00031",
        "cluster_id": "113",
        "generation": "1713028322162524",
        "sha256": "88729f345d9df8614784f82876156cb310debd9599e7614c3bf2a34149897f54",
        "bytes": 127225,
        "dimensions": [768, 1024],
        "rgb_sha256": "56fec3e5ae9ba5fb3d591f3bd6ac92f2daa545c03f67d3861807295209571659",
    },
    {
        "id": "qual_dev_00035",
        "cluster_id": "123",
        "generation": "1713028322227459",
        "sha256": "2dcc0fb62677b796d5f9f879e714185b28b6f9c99099470ed851ab90e46865de",
        "bytes": 54891,
        "dimensions": [1024, 768],
        "rgb_sha256": "bfec94a1511c844e21b28d86d92c6d11dc4f22912d9ff64da3dad349ed937a08",
    },
    {
        "id": "qual_dev_00051",
        "cluster_id": "4",
        "generation": "1713028322800271",
        "sha256": "151387eabb565e0b2b6ef4178c36bc390be683130ed38f3cfa5402797464f6ac",
        "bytes": 140045,
        "dimensions": [1024, 768],
        "rgb_sha256": "693e7a76a07564da6632f071c350a3b79add42d78eb9f58cc2b060c1e1e26307",
    },
    {
        "id": "qual_dev_00056",
        "cluster_id": "112",
        "generation": "1713028322535332",
        "sha256": "dbb0042c2da0d2e9993b3845742abddcefd239b8842af8a961caa7f4bbe07a2a",
        "bytes": 108858,
        "dimensions": [1024, 768],
        "rgb_sha256": "ffe90ffad57c23b04e1671cc63b2043b1dc99e16006f8d8a0055bcc12a71ebdf",
    },
    {
        "id": "qual_dev_00071",
        "cluster_id": "2",
        "generation": "1713028321916922",
        "sha256": "587916bb41f73ba822669abfe394e5ee74756ce83b4d55c5cd3863f618252272",
        "bytes": 141600,
        "dimensions": [1024, 764],
        "rgb_sha256": "871471588bff3a617311c7f3f58763a3a2ddb562c34282b9585f9db6556f62b2",
    },
    {
        "id": "qual_dev_00069",
        "cluster_id": "81",
        "generation": "1713028321833588",
        "sha256": "1e33a59caa9cf2800d2c5e6f3187f565bcf44e6b1e13a9634a4814bf4aaa3e87",
        "bytes": 122687,
        "dimensions": [768, 1024],
        "rgb_sha256": "c7abeadceec404d2a85e4339fc0d811f9a401d3b705c45f28c40f1a35917485c",
    },
]

EVALUATION_CASES = [
    {
        "id": "test_00729",
        "scene_answer": "有人扶著摩托車做特技，摩托車的前輪高高抬起。騎士穿著防護服並戴著安全帽。",
        "facts": [
            {
                "user": "摩托車的前輪是在地上，還是抬起來？",
                "answer": "前輪抬起來，離開地面。",
                "fact_groups": [["抬起", "抬高", "離地", "离地", "raised", "lifted", "wheelie"]],
            },
            {
                "user": "騎士的頭上戴著什麼？",
                "answer": "安全帽。",
                "fact_groups": [["安全帽", "頭盔", "头盔", "helmet"]],
            },
        ],
    },
    {
        "id": "test_01103",
        "scene_answer": "有人握著長桿，把玻璃球伸到發亮的爐口加熱；爐口周圍有金屬支架。",
        "facts": [
            {
                "user": "人正在如何處理長桿前端的玻璃物件？",
                "answer": "把玻璃物件送進爐口加熱。",
                "fact_groups": [["玻璃", "glass"], ["加熱", "加热", "heat", "heating"]],
            },
            {
                "user": "長桿前端的物件被送到哪裡？",
                "answer": "發亮的爐口或加熱爐裡。",
                "fact_groups": [["爐", "炉", "furnace", "kiln"]],
            },
        ],
    },
    {
        "id": "test_00706",
        "scene_answer": "有人用手托著一條淺色、帶橘色斑紋的蛇。蛇身橫跨手背，頭部朝向畫面左側。",
        "facts": [
            {"user": "手上托著的是哪一種動物？", "answer": "蛇。", "fact_groups": [["蛇", "snake"]]},
            {
                "user": "人和這條蛇有什麼接觸關係？",
                "answer": "人用手托著蛇。",
                "fact_groups": [["手", "hand"], ["托", "拿", "抱", "握", "hold", "support"]],
            },
        ],
    },
    {
        "id": "test_00337",
        "scene_answer": "一隻小蜥蜴趴在人的手背與手腕上。背景是模糊的石頭、土壤和綠色植物。",
        "facts": [
            {"user": "照片中央的小動物是什麼？", "answer": "蜥蜴。", "fact_groups": [["蜥蜴", "lizard"]]},
            {
                "user": "蜥蜴是停在人的哪個部位上？",
                "answer": "手腕或手背上。",
                "fact_groups": [["手腕", "手背", "手上", "wrist", "hand"]],
            },
        ],
    },
    {
        "id": "test_00091",
        "scene_answer": "一隻黑棕色小狗趴在窗門前的木地板上。陽光透過窗戶，照在小狗和地板上。",
        "facts": [
            {"user": "小狗正趴在什麼材質的地板上？", "answer": "木地板。", "fact_groups": [["木", "wood"]]},
            {
                "user": "狗是在窗門前趴著，還是在半空跳躍？",
                "answer": "狗趴在窗門前的地板上。",
                "fact_groups": [["趴", "躺", "臥", "卧", "lying", "laying"], ["窗", "門", "门", "window", "door"]],
            },
        ],
    },
    {
        "id": "test_00039",
        "scene_answer": "一輛黑色皮卡停在街邊，後面還排著其他車輛。背景有建築、商店遮棚和交通號誌。",
        "facts": [
            {
                "user": "最前面的黑色車是什麼類型的車？",
                "answer": "皮卡或小貨卡。",
                "fact_groups": [["皮卡", "貨卡", "货卡", "貨車", "货车", "pickup", "truck"]],
            },
            {
                "user": "其他車輛排在這輛皮卡的前面，還是後面？",
                "answer": "後面。",
                "fact_groups": [["後面", "后面", "後方", "后方", "behind"]],
            },
        ],
    },
    {
        "id": "test_00010",
        "scene_answer": "灰白色小鳥棲在粗大的樹幹與樹枝旁，四周有細樹枝、綠葉和藍天。",
        "facts": [
            {
                "user": "圖中的小鳥棲在什麼上面？",
                "answer": "樹枝或樹幹上。",
                "fact_groups": [["樹", "树", "branch", "tree", "trunk"]],
            },
            {"user": "樹枝之間露出的天空是什麼顏色？", "answer": "藍色。", "fact_groups": [["藍", "蓝", "blue"]]},
        ],
    },
    {
        "id": "test_00006",
        "scene_answer": "木桌上散放著尚未拼好的拼圖片，拼圖片印著母雞、小雞和雞舍圖案。",
        "facts": [
            {"user": "木桌上散放著什麼？", "answer": "拼圖片。", "fact_groups": [["拼圖", "拼图", "puzzle", "jigsaw"]]},
            {
                "user": "這些拼圖片已組成完整的一張圖，還是散開、尚未拼好？",
                "answer": "散開，尚未拼好。",
                "fact_groups": [
                    ["散", "未拼", "沒有拼", "没有拼", "分開", "分开", "separate", "scattered", "unassembled", "loose"]
                ],
            },
        ],
    },
    {
        "id": "test_00002",
        "scene_answer": "一組金色雕像立在高台上，包括舉起一隻手的女性與海馬造型。後方是晴朗的藍天。",
        "facts": [
            {
                "user": "這組雕像主要是什麼顏色？",
                "answer": "金色。",
                "fact_groups": [["金色", "金黃", "金黄", "gold", "golden"]],
            },
            {
                "user": "雕像是立在高台上，還是泡在水裡？",
                "answer": "立在高台或石座上。",
                "fact_groups": [
                    ["高台", "台上", "石座", "底座", "紀念碑", "纪念碑", "pedestal", "monument", "platform"]
                ],
            },
        ],
    },
    {
        "id": "test_01629",
        "scene_answer": "路面上有騎自行車的人與車子的影子，也看得到車把、部分車輪和一隻鞋。道路中間有白色標線。",
        "facts": [
            {
                "user": "路面上那個黑色的人形輪廓是什麼？",
                "answer": "騎自行車的人的影子。",
                "fact_groups": [["影子", "陰影", "阴影", "shadow"]],
            },
            {"user": "路面中間的標線是什麼顏色？", "answer": "白色。", "fact_groups": [["白", "white"]]},
        ],
    },
    {
        "id": "test_00094",
        "scene_answer": "黑色表面上貼著一張狗的照片，兩個黃色圓形磁鐵固定照片上緣。照片上還畫有對話泡泡。",
        "facts": [
            {
                "user": "畫面中的狗，是在一張貼起來的照片裡，還是直接站在房間裡？",
                "answer": "在貼起來的照片裡。",
                "fact_groups": [["照片", "相片", "photo", "picture"]],
            },
            {"user": "固定狗照片的圓形磁鐵是什麼顏色？", "answer": "黃色。", "fact_groups": [["黃", "黄", "yellow"]]},
        ],
    },
    {
        "id": "test_00048",
        "scene_answer": "一個老鼠造型的擺飾拿著亮著的燈泡，放在白色架子上，後方可看到電線。",
        "facts": [
            {
                "user": "老鼠造型擺飾拿著什麼？",
                "answer": "燈泡或燈座。",
                "fact_groups": [["燈泡", "灯泡", "燈座", "灯座", "bulb", "lamp"]],
            },
            {
                "user": "這個燈泡目前是亮著，還是熄滅的？",
                "answer": "亮著。",
                "fact_groups": [
                    [
                        "亮著",
                        "亮着",
                        "發光",
                        "发光",
                        "點亮",
                        "点亮",
                        "illuminated",
                        "glowing",
                        "turned on",
                        "light is on",
                        "lamp is on",
                    ]
                ],
            },
        ],
    },
    {
        "id": "qual_dev_00000",
        "scene_answer": "一隻狗玩偶和一隻企鵝玩偶並排坐著，頭上都戴著裝有燈的藍色帽子。後方是窗戶和坐墊。",
        "facts": [
            {
                "user": "狗玩偶右邊坐著哪一種動物玩偶？",
                "answer": "企鵝玩偶。",
                "fact_groups": [["企鵝", "企鹅", "penguin"]],
            },
            {"user": "這兩個玩偶頭上帽子的主要顏色是什麼？", "answer": "藍色。", "fact_groups": [["藍", "蓝", "blue"]]},
        ],
    },
    {
        "id": "qual_dev_00003",
        "scene_answer": "四隻小狗在圍欄內的地毯上，最前面一隻小狗把前腳搭在白色欄邊，旁邊有藍色玩具。",
        "facts": [
            {"user": "照片裡共有幾隻小狗？", "answer": "四隻。", "fact_groups": [["四", "4", "four"]]},
            {
                "user": "最前面的小狗把前腳搭在什麼上面？",
                "answer": "白色欄邊或圍欄上。",
                "fact_groups": [["欄", "栏", "圍板", "围板", "barrier", "fence", "edge"]],
            },
        ],
    },
    {
        "id": "qual_dev_00014",
        "scene_answer": "一尊石獅子雕像前放著大型藍色書本造型物，書封有黃色圓形，後方是樹與建築。",
        "facts": [
            {
                "user": "石獅子前面放著什麼造型的物件？",
                "answer": "書本造型的物件。",
                "fact_groups": [["書", "书", "book"]],
            },
            {"user": "書本造型物的封面主要是什麼顏色？", "answer": "藍色。", "fact_groups": [["藍", "蓝", "blue"]]},
        ],
    },
    {
        "id": "qual_dev_00015",
        "scene_answer": "檯面上放著一個空的金屬網格水果籃，籃子前面有印著 Banana 的小標示牌。",
        "facts": [
            {
                "user": "這個水果籃裡目前裝著水果嗎？",
                "answer": "沒有，籃子是空的。",
                "fact_groups": [["空", "沒有", "没有", "empty", "no fruit"]],
            },
            {
                "user": "水果籃是由實心碗壁，還是金屬網格構成的？",
                "answer": "金屬網格。",
                "fact_groups": [["網格", "网格", "金屬線", "金属线", "wire", "mesh", "lattice"]],
            },
        ],
    },
    {
        "id": "qual_dev_00017",
        "scene_answer": "兩隻貓坐在窗前的木桌上，左邊是灰色貓，右邊是黑白貓；陽光在桌上和貓身上形成條紋。",
        "facts": [
            {"user": "圖中共有幾隻貓坐在桌上？", "answer": "兩隻。", "fact_groups": [["兩", "两", "二", "2", "two"]]},
            {"user": "黑白貓在灰色貓的左邊，還是右邊？", "answer": "右邊。", "fact_groups": [["右", "right"]]},
        ],
    },
    {
        "id": "qual_dev_00021",
        "scene_answer": "有人拿著噴火工具，加熱倒放在支架上的玻璃杯。杯底附近有明亮的火焰。",
        "facts": [
            {
                "user": "手持工具正在對玻璃杯做什麼？",
                "answer": "用火焰加熱玻璃杯。",
                "fact_groups": [["加熱", "加热", "heat"], ["杯", "cup", "mug", "glass"]],
            },
            {
                "user": "照片中的玻璃杯是杯口朝上，還是倒放？",
                "answer": "倒放，杯底朝上。",
                "fact_groups": [["倒放", "倒置", "底朝上", "upside down", "inverted"]],
            },
        ],
    },
    {
        "id": "qual_dev_00031",
        "scene_answer": "有人把紅色水瓶倒過來，讓水從瓶口流向下方的草地，瓶身有白色文字。",
        "facts": [
            {"user": "手上拿著的水瓶主要是什麼顏色？", "answer": "紅色。", "fact_groups": [["紅", "红", "red"]]},
            {
                "user": "人正在用水瓶做什麼？",
                "answer": "把水瓶倒過來，將水倒向草地。",
                "fact_groups": [["倒", "pour", "upside down"], ["水", "water"]],
            },
        ],
    },
    {
        "id": "qual_dev_00035",
        "scene_answer": "一隻灰色貓躺在電視下方的白色平台上，抬頭看著電視；螢幕裡有黑色飛行身影和山景。",
        "facts": [
            {
                "user": "真正的貓位在電視螢幕裡，還是螢幕外？",
                "answer": "螢幕外。",
                "fact_groups": [["螢幕外", "屏幕外", "電視外", "电视外", "outside", "below the tv"]],
            },
            {
                "user": "貓躺的平台是在電視的上方，還是下方？",
                "answer": "下方。",
                "fact_groups": [["下", "below", "under"]],
            },
        ],
    },
    {
        "id": "qual_dev_00051",
        "scene_answer": "一個人的雙腳站在黑色沙灘上，附近有許多小貝殼，海浪泡沫正漫到腳邊。",
        "facts": [
            {
                "user": "沙灘上、腳旁邊有許多什麼小東西？",
                "answer": "貝殼。",
                "fact_groups": [["貝殼", "贝壳", "shell"]],
            },
            {
                "user": "白色的海浪泡沫是遠離雙腳，還是已到腳邊？",
                "answer": "已到腳邊。",
                "fact_groups": [["腳邊", "脚边", "腳上", "脚上", "腳趾", "脚趾", "feet", "toes"]],
            },
        ],
    },
    {
        "id": "qual_dev_00056",
        "scene_answer": "一隻黃黑相間的蝴蝶張開翅膀，停在人的手背上；背景是模糊的草地。",
        "facts": [
            {"user": "停在手背上的昆蟲是什麼？", "answer": "蝴蝶。", "fact_groups": [["蝴蝶", "butterfly"]]},
            {
                "user": "蝴蝶的翅膀是張開，還是收攏？",
                "answer": "張開。",
                "fact_groups": [["張開", "张开", "展開", "展开", "spread", "open", "extended"]],
            },
        ],
    },
    {
        "id": "qual_dev_00071",
        "scene_answer": "三隻孔雀站在道路上，一輛白色皮卡在牠們後方；其中一隻孔雀是白色的。",
        "facts": [
            {"user": "皮卡前面的道路上共有幾隻孔雀？", "answer": "三隻。", "fact_groups": [["三", "3", "three"]]},
            {
                "user": "中間那隻孔雀和旁邊兩隻不同，它主要是什麼顏色？",
                "answer": "白色。",
                "fact_groups": [["白", "white"]],
            },
        ],
    },
    {
        "id": "qual_dev_00069",
        "scene_answer": "窗戶的百葉窗上有人形影子，窗下紅色牆面的架子上展示著幾把刀。",
        "facts": [
            {
                "user": "百葉窗上的人形輪廓是直接看見的人，還是投在窗簾上的影子？",
                "answer": "投在窗簾上的影子。",
                "fact_groups": [["影子", "陰影", "阴影", "shadow"]],
            },
            {
                "user": "展示刀具的架子是在窗戶上方，還是下方？",
                "answer": "下方。",
                "fact_groups": [["下", "below", "under"]],
            },
        ],
    },
]

SOURCE_FILES = [
    {
        "path": "sources/README-docci.md",
        "url": f"https://huggingface.co/datasets/google/docci/raw/{HF_REVISION}/README.md",
        "sha256": "de6b4c24000664c6c580328d32ac70d08801d6210de3178441dba562e28f28ca",
    },
    {
        "path": "sources/docci-loader.py.txt",
        "url": f"https://huggingface.co/datasets/google/docci/raw/{HF_REVISION}/docci.py",
        "sha256": "105b5002a102a9c1a686b2844951b0f86ef9a29e59d6a25d246d2ee943e7f433",
    },
    {
        "path": "sources/docci-website.html",
        "url": f"https://raw.githubusercontent.com/google/docci/{WEBSITE_REVISION}/index.html",
        "sha256": "ddcf7544a01ce1b8069d7def2deb55afe3ab4e301d146b77263011850ee69b44",
    },
    {
        "path": "sources/docci-web-data.jsonl",
        "url": f"https://raw.githubusercontent.com/google/docci/{WEBSITE_REVISION}/web-data/docci_data.jsonlines",
        "sha256": "8f30f1566f9255e07e4b4c5dc2be64f1e70d01a8de486cc6c43b9d5f96d09d4d",
    },
    {
        "path": "sources/docci-descriptions.jsonl",
        "url": "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"
        f"?generation={DESCRIPTION_GENERATION}",
        "sha256": DESCRIPTION_SHA256,
    },
]

LICENSE_NOTICE = """DOCCI images and original human-authored descriptions

Copyright and licensing attribution: Google LLC.
Images taken by Jason Baldridge and his family; descriptions by DOCCI human annotators.
Authors: Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho,
Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer,
Su Wang, Jason Baldridge. DOCCI: Descriptions of Connected and Contrasting Images,
ECCV 2024. Paper: https://arxiv.org/abs/2404.19753

Images and annotations: Creative Commons Attribution 4.0 International (CC BY 4.0).
License: https://creativecommons.org/licenses/by/4.0/
Legal code: https://creativecommons.org/licenses/by/4.0/legalcode
Official statement: https://google.github.io/docci/ (snapshot included in sources).
CC BY 4.0 allows reuse, redistribution and adaptation, including commercial use,
subject to attribution, a license link, and indication of changes. This does not
mean every third-party right in depicted material has been independently examined.
These external data do not inherit this repository's MIT license.

Adaptation by the Tiny Perceptron educational project:
- use publisher-supplied JPEG thumbnails, not the original-resolution image archive;
- select 120 official train, 12 qual_dev and 12 official test images;
- use only the first complete English caption sentence as a training target;
- add manually AI-authored Traditional Chinese validation/test questions and answers;
- retain full original captions, source revisions, per-image GCS generations and hashes.
Adapted questions, answers, and this combined data selection use CC BY 4.0.
Training examples were not generated by a vision model or translated automatically.
Test questions and answers were frozen after AI image inspection and before inference.
"""

LIMITATIONS = [
    "This is a small teaching subset, not evidence of general scene understanding.",
    "Official DOCCI images are related by design; similarity clusters are k-means groups, not capture sessions.",
    "All 149 upstream clusters cross official train/test; image identity separation does not ensure entity separation.",
    "The chosen base model's pretraining data are not fully disclosed; these images may already have been seen.",
    "Exact file SHA256 and decoded RGB hashes are separated across splits; near-duplicates are not proven absent.",
    "Training targets are original English captions; Chinese held-out questions also test transferred language skills.",
    "Scene answers require manual review; keyword fact rubrics are limited proxies, not full semantic judges.",
    "Some official captions simplify or miss visible details; added answers use conservative visible facts.",
    "DOCCI is geographically and subject-biased; faces are often removed or blurred and people-action coverage is limited.",
]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def ensure_download(spec, destination):
    """Check content even when cached; never silently accept a changed upstream file."""
    if destination.exists():
        data = destination.read_bytes()
        if sha256_bytes(data) != spec["sha256"]:
            raise ValueError(f"Cached source hash mismatch: {destination}")
        return data
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "TinyPerceptronDOCCI/3"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if sha256_bytes(data) != spec["sha256"]:
                raise ValueError(f"Downloaded source hash mismatch: {spec['url']}")
            destination.write_bytes(data)
            return data
        except (OSError, ValueError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise ValueError(f"Unable to fetch pinned source: {spec['url']}: {last_error}")


def image_source(spec):
    return {
        **spec,
        "url": f"https://storage.googleapis.com/docci/thumbnails/{spec['id']}.jpg?generation={spec['generation']}",
        "path": f"images/{spec['id']}.jpg",
        "image_variant": "publisher thumbnail",
    }


def decoded_image_digest(path):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return list(rgb.size), sha256_bytes((str(rgb.size) + "RGB").encode() + rgb.tobytes())


def first_caption_sentence(description):
    return re.split(r"(?<=[.!?])\s+", description.strip(), maxsplit=1)[0]


def original_records(directory):
    return {
        record["example_id"]: record
        for record in map(json.loads, (directory / "sources/docci-descriptions.jsonl").read_text().splitlines())
    }


def build_manifest(directory):
    originals = original_records(directory)
    images = {spec["id"]: image_source(spec) for spec in IMAGE_SPECS}
    rows = []
    selected_originals = []
    for image_id, image in images.items():
        original = originals[image_id]
        selected_originals.append({**original, "image": image})
    for image_id in TRAIN_IDS:
        original = originals[image_id]
        if original["split"] != "train":
            raise ValueError(f"Training image is not in official train: {image_id}")
        rows.append(
            {
                "id": f"docci/{image_id}/caption",
                "split": "train",
                "task": "scene",
                "family": f"docci:{image_id}",
                "user": "Describe the main scene and what the visible subjects are doing. Use one sentence.",
                "answer": first_caption_sentence(original["description"]),
                "image": images[image_id]["path"],
                "source": {
                    "repo": "google/docci",
                    "revision": HF_REVISION,
                    "original_id": image_id,
                    "official_split": original["split"],
                    "license": "CC BY 4.0",
                    "description_source_sha256": DESCRIPTION_SHA256,
                    "image_generation": images[image_id]["generation"],
                    "image_sha256": images[image_id]["sha256"],
                },
            }
        )
    for case in EVALUATION_CASES:
        image_id = case["id"]
        original = originals[image_id]
        split = "validation" if original["split"] == "qual_dev" else "test"
        if original["split"] not in {"qual_dev", "test"}:
            raise ValueError(f"Evaluation image is not in the assigned official split: {image_id}")
        shared = {
            "split": split,
            "task": "scene",
            "family": f"docci:{image_id}",
            "image": images[image_id]["path"],
            "source": {
                "repo": "google/docci",
                "revision": HF_REVISION,
                "original_id": image_id,
                "official_split": original["split"],
                "license": "CC BY 4.0",
                "description_source_sha256": DESCRIPTION_SHA256,
                "image_generation": images[image_id]["generation"],
                "image_sha256": images[image_id]["sha256"],
                "target_author": "AI reader after direct image inspection and complete official caption reading",
                "target_frozen_before_model_inference": True,
            },
        }
        rows.append(
            {
                **shared,
                "id": f"docci/{image_id}/scene",
                "user": "請用繁體中文描述這張照片的主要內容，以及物件或人物之間的關係。",
                "answer": case["scene_answer"],
                "references": {
                    "kind": "manual",
                    "reference_answer": case["scene_answer"],
                    "rubric": "Read the actual image and full original caption; check visible subjects, actions and relations; "
                    "do not require identical wording or invent events outside the frame.",
                    "official_description": original["description"],
                },
            }
        )
        for position, fact in enumerate(case["facts"], start=1):
            rows.append(
                {
                    **shared,
                    "id": f"docci/{image_id}/fact-{position}",
                    "user": fact["user"],
                    "answer": fact["answer"],
                    "references": {
                        "kind": "facts",
                        "fact_groups": fact["fact_groups"],
                        "forbidden": [],
                        "rubric_limit": "OR within a fact group, AND across groups; keyword matching cannot judge "
                        "negation or all semantically equivalent answers; retain outputs for manual inspection.",
                    },
                }
            )
    return {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "sources": [
            {
                "dataset": "DOCCI",
                "repo": "google/docci",
                "revision": HF_REVISION,
                "website_revision": WEBSITE_REVISION,
                "license": "CC BY 4.0",
                "image_creator": "Jason Baldridge and family",
                "licensor": "Google LLC",
                "annotation_creator": "human annotators",
                "original_language": "English",
                "description_generation": DESCRIPTION_GENERATION,
                "description_sha256": DESCRIPTION_SHA256,
                "repository_revision_scope": "README and loader only; external annotation and images have separate GCS pins",
                "files": SOURCE_FILES,
            }
        ],
        "selection": {
            "train_images": 120,
            "validation_images": 12,
            "test_images": 12,
            "train_strategy": "One official train image per each of 120 k-means clusters, ranked by SHA256 of "
            "'tiny-perceptron-natural-vision-v3:' + cluster_id and then example_id within each cluster.",
            "validation_strategy": "12 preselected official qual_dev images, selected for scene diversity before inference",
            "test_strategy": "12 preselected official test images, selected for scene/action/relation diversity before inference",
            "training_target": "First complete sentence of original human English caption; no generated model labels",
            "split_unit": "source image identity; all questions of the same photo remain in its split",
            "test_is_not_training": True,
        },
        "images": list(images.values()),
        "limitations": LIMITATIONS,
        "rows": rows,
    }, selected_originals


def write_dataset_files(directory):
    manifest, originals = build_manifest(directory)
    (directory / "manifest.json").write_bytes(json_bytes(manifest))
    (directory / "selected-descriptions.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in originals), encoding="utf-8"
    )
    (directory / "LICENSE-DOCCI.txt").write_text(LICENSE_NOTICE, encoding="utf-8")
    (directory / "ATTRIBUTION.json").write_bytes(
        json_bytes(
            {
                "dataset": "DOCCI",
                "licensor": "Google LLC",
                "image_creator": "Jason Baldridge and family",
                "caption_creator": "DOCCI human annotators",
                "citation": "Yasumasa Onoe et al. DOCCI: Descriptions of Connected and Contrasting Images. ECCV 2024.",
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "homepage": "https://google.github.io/docci/",
                "modifications": "Publisher thumbnails; subset selection; first-caption-sentence targets; "
                "AI-authored held-out Traditional Chinese questions and conservative references after image inspection.",
                "image_attributions": [
                    {"original_id": image["id"], "source_url": image["url"], "sha256": image["sha256"]}
                    for image in manifest["images"]
                ],
            }
        )
    )
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "checksums.json")
    (directory / "checksums.json").write_bytes(
        json_bytes(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": str(path.relative_to(directory)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_bytes(path.read_bytes()),
                    }
                    for path in paths
                ],
            }
        )
    )


def validate(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    expected, _ = build_manifest(directory)
    if manifest != expected:
        raise ValueError("Manifest differs from the frozen source selection and inspected question set")
    identities = collections.defaultdict(set)
    digest_splits = collections.defaultdict(set)
    rgb_splits = collections.defaultdict(set)
    row_ids = set()
    image_specs = {spec["path"]: spec for spec in manifest["images"]}
    for row in manifest["rows"]:
        if row["id"] in row_ids:
            raise ValueError("Duplicate row ID")
        row_ids.add(row["id"])
        identities[row["family"]].add(row["split"])
        image = image_specs[row["image"]]
        digest_splits[image["sha256"]].add(row["split"])
        rgb_splits[image["rgb_sha256"]].add(row["split"])
    if any(len(splits) > 1 for table in [identities, digest_splits, rgb_splits] for splits in table.values()):
        raise ValueError("Image identity or exact file/pixel duplicates cross data splits")
    for spec in SOURCE_FILES:
        if sha256_bytes((directory / spec["path"]).read_bytes()) != spec["sha256"]:
            raise ValueError(f"Source hash mismatch: {spec['path']}")
    for spec in manifest["images"]:
        path = directory / spec["path"]
        if path.stat().st_size != spec["bytes"] or sha256_bytes(path.read_bytes()) != spec["sha256"]:
            raise ValueError(f"Image file hash/size mismatch: {spec['path']}")
        dimensions, rgb_digest = decoded_image_digest(path)
        if dimensions != spec["dimensions"] or rgb_digest != spec["rgb_sha256"]:
            raise ValueError(f"Image decoded pixel mismatch: {spec['path']}")
    checksums = json.loads((directory / "checksums.json").read_text())
    for spec in checksums["files"]:
        path = directory / spec["path"]
        if path.stat().st_size != spec["bytes"] or sha256_bytes(path.read_bytes()) != spec["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {spec['path']}")
    counts = dict(collections.Counter(row["split"] for row in manifest["rows"]))
    if counts != {"train": 120, "validation": 36, "test": 36}:
        raise ValueError(f"Unexpected row counts: {counts}")
    return {
        "dataset_version": DATASET_VERSION,
        "rows": counts,
        "images": len(manifest["images"]),
        "image_bytes": sum(spec["bytes"] for spec in manifest["images"]),
        "manifest_sha256": sha256_bytes((directory / "manifest.json").read_bytes()),
        "cross_split_identity_or_exact_pixel_duplicates": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/natural/vision"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    directory = args.output
    if not args.validate_only:
        directory.mkdir(parents=True, exist_ok=True)
        for spec in SOURCE_FILES:
            ensure_download(spec, directory / spec["path"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(
                executor.map(
                    lambda spec: ensure_download(image_source(spec), directory / image_source(spec)["path"]),
                    IMAGE_SPECS,
                )
            )
        write_dataset_files(directory)
    print(json.dumps(validate(directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
