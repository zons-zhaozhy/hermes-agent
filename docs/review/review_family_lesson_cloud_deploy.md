# 反方审查报告：家庭课件部署到云服务器供孩子在家用

审查日期：2026-09-08 · 性质：只找漏洞，不含实施方案
证据基线：make_launcher.py / make_lesson.py / make_practice.py（loom/scenarios/math_to_manim/）、deploy/nginx_gateway.conf、deploy/docker-compose.yml、skill: teaching-content-generator / ontox-deploy-channel-selection

## 一、安全

| # | 问题 | 风险 |
|---|------|------|
| S1 | 公网无鉴权=儿童内容全量可爬。gateway 鉴权手段仅 auth_request /_aml_admin_check（nginx_gateway.conf L562/596/605，只护 AML API）；静态路径零鉴权。入口页 学习入口.html 明文链接全部课件，拿到一个 URL 即可枚举全部内容。 | 高 |
| S2 | HTTP 80 明文通道仍开着（nginx_gateway.conf:230-232 注释明说 301 跳转已移除），无 HSTS 兜底。 | 高 |
| S3 | 入口页泄露家庭结构：「👦哥哥的/👧妹妹的」两栏（make_launcher.py:89-90）+ PROFILES 内嵌 14岁/10岁画像（make_lesson.py）。未成年人性别、年龄、学习进度对陌生人可读。PIPL 下不满14周岁未成年人信息属敏感个人信息，无鉴权公开是合规硬伤。 | 高 |
| S4 | 云端囤课=凭据上云：SERVICE_JWT、qfwy_supplier 的 LLM 供应商凭据都要在公网 VPS 可用，暴露面从家里一台 Mac 扩大到 VPS。 | 中 |

## 二、部署通道

| # | 问题 | 风险 |
|---|------|------|
| D1 | 主 server 块静态内容烧在镜像里（root /usr/share/nginx/html，Dockerfile COPY static/）。存在免重建通道：nginx_gateway.conf:717 include /etc/nginx/domains/*.conf + ./certs/domains ro 挂载（compose L788），可 drop-in 新 server 块；但 ①certs/domains/ 当前为空=通道从未用过；②课件内容卷仍需改 docker-compose.yml（gateway 只挂 certs 目录）；③改动游离在 build.sh --list 服务清单之外，ontox-doctor 对账是否视为漂移未验证。 | 中-高 |
| D2 | 手工挂卷/改 conf 与 doctor 自愈体系的漂移判定冲突，方案未交代白名单机制。 | 中 |
| D3 | 对象存储+CDN 是第三套运维面：回源鉴权、桶公开读误配、无 doctor 巡检，与"复用现有体系"初衷矛盾。 | 中 |
| D4 | 入口刷新链路断裂：make_launcher 的运维契约是"家长本机重跑刷新"（make_launcher.py:104，扫 ~/Lessons）。云端囤课后谁触发 launcher 重生成、产物如何进托管位置，无 owner。 | 中 |
| D5 | storefront 通道前车之鉴：docker-compose.yml:710-712 注释实录"容器重建抹掉 .output 层 → DB 里 /generated/* URL 全 404"，需双挂卷抢救。任何"新容器托管课件"方案都面对同类容器层内容易失问题。 | 中 |

## 三、手机体验

| # | 问题 | 风险 |
|---|------|------|
| M1 | 桌面视口假设未实测：canvas 固定 420x300 无 dpr 缩放；入口页 .col min-width:300px（make_launcher.py:77）在 375px 屏必横滚；课件 body max-width:900px、无任何 @media 断点。 | 中 |
| M2 | 大 base64 音频全嵌单文件，手机端 atob+decodeAudioData 懒解码在低端安卓/iOS 微信内置浏览器有内存风险（skill 已实录 file:// 下大 data URI 哑巴）。 | 中 |
| M3 | 错题本 localStorage 不跨设备（make_practice.py:151-161）：手机练的错题电脑看不到，无痕模式即清零，多设备场景"只练错的"失效。 | 低-中 |

## 四、运维

| # | 问题 | 风险 |
|---|------|------|
| O1 | 证书自动化存在（storefront-cert-agent: Let's Encrypt webroot + docker.sock reload gateway，compose L805-832）但三个缺口：①LETSENCRYPT_EMAIL 默认空值，.env 未配则 agent 形同虚设；②主 conf 用的 ontoxai.com.pem（L246-247）是否在 agent 管辖域内未证实；③cert-agent 自身是 cloud profile 服务，它死了续期静默停止。 | 中 |
| O2 | 服务器续费/实例到期=课件全灭，且方案触发场景恰是长期出差，本地 ~/Lessons 若清掉只留云端，单点在一张账单上。 | 中-高 |
| O3 | 囤课失败无人知：TTS 间歇失败/Cortex 没跑/LLM 配额，只写云端日志，孩子看到的是空入口页。 | 中 |

## 五、隐私

| # | 问题 | 风险 |
|---|------|------|
| P1 | 孩子学习行为进 nginx access log（时间/IP/学哪课）；走 CDN 则再出域第三方。内容本身不上传后端（纯静态+localStorage 不外发，这点方案是对的）。 | 低 |
| P2 | 云端 cron 自动囤课把"孩子年级/薄弱点送第三方 LLM"常态化、无人审放大；错题回灌会让画像越攒越细。 | 中 |

## 结论

最脆弱三柱：① "gateway 静态托管零成本复用"前提部分不成立——免重建通道存在但从未启用且仍需改 compose（D1/D5）；② 无鉴权+HTTP明文+儿童画像三重叠加（S1-S3），合规硬伤不可对冲；③ 证书续期缺口+服务器账单单点（O1/O2），恰在家长长期不在家时爆发。手机体验（M1-M3）纯属未验证假设。

## 审查中的自我修正（相对初审的两处降级）

1. O1 初判"无证书自动续期证据"（高）→ 复核 docker-compose.yml 发现 storefront-cert-agent 存在，降为中，保留三个残余缺口（email 空值/域管辖未证/agent 自身单点）。
2. D1 初判"每课必须重打镜像"（高）→ 复核发现 nginx_gateway.conf:717 include /etc/nginx/domains/*.conf 免重建 drop-in 通道，降为中-高：通道存在但 certs/domains 为空（未启用）、内容卷仍需改 compose。
