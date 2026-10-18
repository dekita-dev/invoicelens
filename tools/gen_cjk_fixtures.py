"""Generate a synthetic CJK invoice fixture set with exact ground truth.

Why synthetic first
-------------------
Every public invoice dataset I know of is either Latin-script (CORD, SROIE,
FUNSD) or has redistribution terms I cannot put in a repo that ships to
customers (XFUND is fine, but its Japanese subset is forms, not invoices).
A rendered set costs nothing, ships with the repo, and every ground-truth
character is known by construction -- there is no annotation error, so a low
score can only mean the engine is wrong, never the labels.

Tiers
-----
  clean : black text on white, no rotation, no noise. Establishes an upper
          bound. If an engine fails here it will certainly fail in the wild.
  scan  : mild blur + gaussian noise + 1.5 degree rotation + JPEG artifacts.
          Approximates a flatbed scan of a printed invoice.
  photo : perspective warp onto a dark desk + uneven light + JPEG. This is
          the phone-photo case the preprocessing pass exists for: no page
          border is straight, the light is a gradient, the page fills only
          part of the frame.
  hard  : photo + downscaled to ~55% + heavier blur. A cheap phone in bad
          light. If the total survives this, it survives a real pocket.

Usage
-----
  python tools/gen_cjk_fixtures.py                 # all tiers, all langs
  python tools/gen_cjk_fixtures.py --tier clean     # just the easy tier
  python tools/gen_cjk_fixtures.py --lang ja        # one language
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import zlib

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------
# Fonts. All four are shipped with Windows; the script refuses to guess.
# A missing font is a hard error, because silently substituting a Latin font
# would produce images that look like a CJK test but contain no CJK glyphs.
# --------------------------------------------------------------------------
FONT_DIR = r"C:\Windows\Fonts"
FONTS = {
    # lang: (file, ttc index, label shown in the report)
    "zh-Hans": ("msyh.ttc", 0, "Microsoft YaHei"),
    "zh-Hant": ("msyh.ttc", 0, "Microsoft YaHei (Hant coverage)"),
    "ja": ("msgothic.ttc", 0, "MS Gothic"),
    "ko": ("malgun.ttf", 0, "Malgun Gothic"),
    # English is here as the control, not as a target: if the engine cannot
    # clear a Latin-script invoice, a CJK number is meaningless, because the
    # difference between them would then be measuring something else.
    "en": ("arial.ttf", 0, "Arial"),
}

# --------------------------------------------------------------------------
# Fixture definitions. Hand-written, not random: a random generator would
# drift toward whatever the engine happens to read well. Currency, wording and
# date format are the ones each locale actually prints.
# --------------------------------------------------------------------------
INVOICES = [
    {
        "id": "hans-design-retainer",
        "lang": "zh-Hans",
        "vendor": "上海云帆设计工作室",
        "address": "上海市徐汇区漕溪北路 88 号 12 楼",
        "doc_title": "增值税普通发票",
        "doc_no_label": "发票号码",
        "doc_no": "04512026",
        "date_label": "开票日期",
        "date": "2026年09月28日",
        "col_headers": ["项目名称", "数量", "单价", "金额"],
        "items": [
            ["品牌视觉设计服务", "1", "8,000.00", "8,000.00"],
            ["包装设计修改", "3", "1,200.00", "3,600.00"],
        ],
        "subtotal_label": "小计",
        "subtotal": "11,600.00",
        "tax_label": "税额（6%）",
        "tax": "696.00",
        "total_label": "价税合计",
        "total": "12,296.00",
        "currency": "CNY",
    },
    {
        "id": "hans-software-renewal",
        "lang": "zh-Hans",
        "vendor": "杭州数擎科技有限公司",
        "address": "浙江省杭州市余杭区文一西路 969 号",
        "doc_title": "电子发票",
        "doc_no_label": "发票代码",
        "doc_no": "033002100511",
        "date_label": "开票日期",
        "date": "2026年10月06日",
        "col_headers": ["服务名称", "数量", "单价", "金额"],
        "items": [
            ["企业版年度订阅", "1", "4,980.00", "4,980.00"],
            ["数据导出模块", "2", "399.00", "798.00"],
        ],
        "subtotal_label": "小计",
        "subtotal": "5,778.00",
        "tax_label": "税额（6%）",
        "tax": "346.68",
        "total_label": "价税合计",
        "total": "6,124.68",
        "currency": "CNY",
    },
    {
        # A third template per language, so a language score is not one
        # document's luck. This one is a freight invoice: 9% tax (China's
        # transport rate) instead of the 6% used elsewhere, which also keeps
        # the arithmetic check from being satisfied by a repeated number.
        "id": "hans-logistics",
        "lang": "zh-Hans",
        "vendor": "广州顺捷物流有限公司",
        "address": "广东省广州市白云区机场路 1023 号",
        "doc_title": "增值税专用发票",
        "doc_no_label": "发票号码",
        "doc_no": "05277143",
        "date_label": "开票日期",
        "date": "2026年10月09日",
        "col_headers": ["项目名称", "数量", "单价", "金额"],
        "items": [
            ["陆运运费（上海—广州）", "2", "3,200.00", "6,400.00"],
            ["装卸及仓储费", "1", "860.00", "860.00"],
        ],
        "subtotal_label": "小计",
        "subtotal": "7,260.00",
        "tax_label": "税额（9%）",
        "tax": "653.40",
        "total_label": "价税合计",
        "total": "7,913.40",
        "currency": "CNY",
    },
    {
        "id": "hant-translation",
        "lang": "zh-Hant",
        "vendor": "台北澄語翻譯社",
        "address": "台北市大安區復興南路一段 390 號 5 樓",
        "doc_title": "統一發票",
        "doc_no_label": "發票號碼",
        "doc_no": "AB-48213907",
        "date_label": "開立日期",
        "date": "2026年09月25日",
        "col_headers": ["品名", "數量", "單價", "金額"],
        "items": [
            ["技術文件英譯中", "12", "1,500.00", "18,000.00"],
            ["術語表建置", "1", "3,500.00", "3,500.00"],
        ],
        "subtotal_label": "小計",
        "subtotal": "21,500.00",
        "tax_label": "營業稅（5%）",
        "tax": "1,075.00",
        "total_label": "總計",
        "total": "22,575.00",
        "currency": "TWD",
    },
    {
        "id": "hant-photo-studio",
        "lang": "zh-Hant",
        "vendor": "台中光合影像工作室",
        "address": "台中市西區公益路 168 號 3 樓",
        "doc_title": "收據",
        "doc_no_label": "單據編號",
        "doc_no": "PT-2026-0928",
        "date_label": "開立日期",
        "date": "2026年09月28日",
        "col_headers": ["項目", "數量", "單價", "金額"],
        "items": [
            ["商品攝影半日", "1", "9,000.00", "9,000.00"],
            ["後製修圖", "20", "450.00", "9,000.00"],
        ],
        "subtotal_label": "小計",
        "subtotal": "18,000.00",
        "tax_label": "營業稅（5%）",
        "tax": "900.00",
        "total_label": "總計",
        "total": "18,900.00",
        "currency": "TWD",
    },
    {
        # Catering: many small quantities and a 5% business tax, the shape a
        # restaurant receipt actually has. Also the only template whose line
        # items both end in .00, which is where a sloppy amount parser starts
        # treating the decimal point as a thousands separator.
        "id": "hant-catering",
        "lang": "zh-Hant",
        "vendor": "高雄好食餐飲有限公司",
        "address": "高雄市前鎮區中山二路 260 號",
        "doc_title": "統一發票",
        "doc_no_label": "發票號碼",
        "doc_no": "CD-77120458",
        "date_label": "開立日期",
        "date": "2026年10月07日",
        "col_headers": ["品名", "數量", "單價", "金額"],
        "items": [
            ["會議餐盒", "40", "180.00", "7,200.00"],
            ["茶飲外送", "40", "55.00", "2,200.00"],
        ],
        "subtotal_label": "小計",
        "subtotal": "9,400.00",
        "tax_label": "營業稅（5%）",
        "tax": "470.00",
        "total_label": "總計",
        "total": "9,870.00",
        "currency": "TWD",
    },
    {
        "id": "ja-consulting",
        "lang": "ja",
        "vendor": "株式会社みなと総合研究所",
        "address": "東京都港区芝浦三丁目 12 番 5 号",
        "doc_title": "請求書",
        "doc_no_label": "請求書番号",
        "doc_no": "INV-2026-0928",
        "date_label": "発行日",
        "date": "2026年09月28日",
        "col_headers": ["品目", "数量", "単価", "金額"],
        "items": [
            ["業務改善コンサルティング", "2", "120,000", "240,000"],
            ["現地調査費用", "1", "48,000", "48,000"],
        ],
        "subtotal_label": "小計",
        "subtotal": "288,000",
        "tax_label": "消費税（10%）",
        "tax": "28,800",
        "total_label": "合計",
        "total": "316,800",
        "currency": "JPY",
    },
    {
        "id": "ja-printing",
        "lang": "ja",
        "vendor": "有限会社かわせみ印刷",
        "address": "大阪府大阪市中央区谷町二丁目 4 番 11 号",
        "doc_title": "請求書",
        "doc_no_label": "請求書番号",
        "doc_no": "KW-2609-0142",
        "date_label": "発行日",
        "date": "2026年09月30日",
        "col_headers": ["品目", "数量", "単価", "金額"],
        "items": [
            ["会社案内パンフレット印刷", "500", "180", "90,000"],
            ["製版代", "1", "15,000", "15,000"],
        ],
        "subtotal_label": "小計",
        "subtotal": "105,000",
        "tax_label": "消費税（10%）",
        "tax": "10,500",
        "total_label": "合計",
        "total": "115,500",
        "currency": "JPY",
    },
    {
        # Parts order: three- and six-digit unit prices with no decimals, the
        # Japanese convention, and a document number that mixes letters and
        # digits in the middle rather than at the front.
        "id": "ja-parts",
        "lang": "ja",
        "vendor": "株式会社ヤマト精密部品",
        "address": "愛知県名古屋市中村区名駅四丁目 7 番 1 号",
        "doc_title": "請求書",
        "doc_no_label": "請求書番号",
        "doc_no": "YP-2026-1007",
        "date_label": "発行日",
        "date": "2026年10月07日",
        "col_headers": ["品目", "数量", "単価", "金額"],
        "items": [
            ["精密軸受ユニット", "120", "4,500", "540,000"],
            ["検査治具", "1", "86,000", "86,000"],
        ],
        "subtotal_label": "小計",
        "subtotal": "626,000",
        "tax_label": "消費税（10%）",
        "tax": "62,600",
        "total_label": "合計",
        "total": "688,600",
        "currency": "JPY",
    },
    {
        "id": "ko-development",
        "lang": "ko",
        "vendor": "주식회사 한빛소프트웍스",
        "address": "서울특별시 마포구 월드컵북로 396",
        "doc_title": "세금계산서",
        "doc_no_label": "계산서번호",
        "doc_no": "2026-09-28-0041",
        "date_label": "작성일자",
        "date": "2026년 09월 28일",
        "col_headers": ["품목", "수량", "단가", "금액"],
        "items": [
            ["웹 애플리케이션 개발", "1", "6,500,000", "6,500,000"],
            ["유지보수 3개월", "3", "450,000", "1,350,000"],
        ],
        "subtotal_label": "공급가액",
        "subtotal": "7,850,000",
        "tax_label": "부가세（10%）",
        "tax": "785,000",
        "total_label": "합계금액",
        "total": "8,635,000",
        "currency": "KRW",
    },
    {
        "id": "ko-translation",
        "lang": "ko",
        "vendor": "번역공방 서울",
        "address": "서울특별시 종로구 종로 51",
        "doc_title": "청구서",
        "doc_no_label": "청구번호",
        "doc_no": "TR-2026-1006",
        "date_label": "작성일자",
        "date": "2026년 10월 06일",
        "col_headers": ["품목", "수량", "단가", "금액"],
        "items": [
            ["계약서 한영 번역", "24", "35,000", "840,000"],
            ["감수 비용", "1", "120,000", "120,000"],
        ],
        "subtotal_label": "공급가액",
        "subtotal": "960,000",
        "tax_label": "부가세（10%）",
        "tax": "96,000",
        "total_label": "합계금액",
        "total": "1,056,000",
        "currency": "KRW",
    },
    {
        # Marketing retainer: two line items that both round to whole won,
        # so the amount strings carry no decimal point at all.
        "id": "ko-marketing",
        "lang": "ko",
        "vendor": "주식회사 파도마케팅",
        "address": "서울특별시 강남구 테헤란로 152",
        "doc_title": "세금계산서",
        "doc_no_label": "계산서번호",
        "doc_no": "2026-10-07-0112",
        "date_label": "작성일자",
        "date": "2026년 10월 07일",
        "col_headers": ["품목", "수량", "단가", "금액"],
        "items": [
            ["퍼포먼스 광고 운영", "1", "3,200,000", "3,200,000"],
            ["콘텐츠 제작", "4", "380,000", "1,520,000"],
        ],
        "subtotal_label": "공급가액",
        "subtotal": "4,720,000",
        "tax_label": "부가세（10%）",
        "tax": "472,000",
        "total_label": "합계금액",
        "total": "5,192,000",
        "currency": "KRW",
    },
    # ------------------------------------------------------------------
    # English, as the control group. Three distinct shapes on purpose:
    # a services invoice with a percentage tax that lands on cents
    # (.25), a goods invoice whose tax lands on .96, and a SaaS invoice
    # whose tax lands on .85. A single template would let a systematic
    # decimal-point failure hide behind one lucky amount.
    # ------------------------------------------------------------------
    {
        "id": "en-legal",
        "lang": "en",
        "vendor": "Harbor & Vale LLP",
        "address": "2200 Westlake Avenue, Suite 400, Seattle, WA 98121",
        "doc_title": "INVOICE",
        "doc_no_label": "Invoice No.",
        "doc_no": "HV-2026-1042",
        "date_label": "Invoice Date",
        "date": "10/07/2026",
        "col_headers": ["Description", "Qty", "Unit Price", "Amount"],
        "items": [
            ["Contract review and redlines", "6", "450.00", "2,700.00"],
            ["Filing fee (pass-through)", "1", "320.00", "320.00"],
        ],
        "subtotal_label": "Subtotal",
        "subtotal": "3,020.00",
        "tax_label": "Sales Tax (8.75%)",
        "tax": "264.25",
        "total_label": "Total Due",
        "total": "3,284.25",
        "currency": "USD",
    },
    {
        "id": "en-hardware",
        "lang": "en",
        "vendor": "Northline Supply Co.",
        "address": "4180 Industrial Parkway, Columbus, OH 43228",
        "doc_title": "INVOICE",
        "doc_no_label": "Invoice No.",
        "doc_no": "NS-88213",
        "date_label": "Invoice Date",
        "date": "10/09/2026",
        "col_headers": ["Item", "Qty", "Unit Price", "Amount"],
        "items": [
            ["Rackmount server rails", "4", "129.00", "516.00"],
            ["10G SFP+ transceiver", "8", "74.50", "596.00"],
        ],
        "subtotal_label": "Subtotal",
        "subtotal": "1,112.00",
        "tax_label": "Sales Tax (8.00%)",
        "tax": "88.96",
        "total_label": "Total Due",
        "total": "1,200.96",
        "currency": "USD",
    },
    {
        "id": "en-subscription",
        "lang": "en",
        "vendor": "Tallybase Inc.",
        "address": "55 Water Street, Floor 12, New York, NY 10041",
        "doc_title": "INVOICE",
        "doc_no_label": "Invoice No.",
        "doc_no": "TB-2026-1001",
        "date_label": "Invoice Date",
        "date": "10/01/2026",
        "col_headers": ["Description", "Qty", "Unit Price", "Amount"],
        "items": [
            ["Team plan, annual (12 seats)", "12", "240.00", "2,880.00"],
            ["Priority support add-on", "1", "600.00", "600.00"],
        ],
        "subtotal_label": "Subtotal",
        "subtotal": "3,480.00",
        "tax_label": "Sales Tax (8.875%)",
        "tax": "308.85",
        "total_label": "Total Due",
        "total": "3,788.85",
        "currency": "USD",
    },
]


def check_template_arithmetic() -> None:
    """Stop before rendering if any template's numbers do not add up.

    subtotal + tax == total is the corpus's second scoring axis, so a
    template that is off by a cent would mark every page built from it as
    a failure and the report would blame the OCR engine for a typo here.
    Failing at generation time, with the invoice id in the message, keeps
    the two failure modes apart.
    """
    from decimal import Decimal, InvalidOperation

    bad = []
    for inv in INVOICES:
        try:
            sub, tax, total = (Decimal(inv[k].replace(",", ""))
                               for k in ("subtotal", "tax", "total"))
        except (InvalidOperation, KeyError) as exc:
            bad.append((inv["id"], "unparseable amount (%s)" % exc))
            continue
        if sub + tax != total:
            bad.append((inv["id"], "%s + %s != %s" % (sub, tax, total)))
    if bad:
        raise SystemExit("FATAL: template arithmetic is wrong: %s" % bad)


W, H = 780, 900
MARGIN = 56


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    full = os.path.join(FONT_DIR, path)
    if not os.path.isfile(full):
        raise SystemExit("FATAL: font not found: %s" % full)
    return ImageFont.truetype(full, size)


# The amount column shares its right edge with the totals below it (W - MARGIN),
# which is how a real invoice is laid out and what makes a misread column
# obvious to a human reading the image.
COL_X = [MARGIN, 452, 588, W - MARGIN]
COL_ALIGN = ["left", "right", "right", "right"]


def _solve8(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination for the 8x8 perspective system. Pure stdlib:
    the fixture generator must stay runnable anywhere PIL runs."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular perspective system")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            if f:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def _find_coeffs(pa, pb) -> list[float]:
    """Coefficients for Image.PERSPECTIVE.

    PIL's transform walks the OUTPUT image and asks, for each pixel, where
    to fetch from the source -- so the coefficients are the map from output
    space to input space. That makes the argument order the inverse of the
    intuitive one: pa is the DESTINATION quad (in output space), pb is the
    SOURCE quad (in the page). Verified empirically with corner markers:
    get it backwards and the page silently lands in the wrong place, which
    still looks like a photo and is still a broken fixture.
    """
    rows = []
    for (x1, y1), (x2, y2) in zip(pa, pb):
        rows.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])
        rows.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1])
    return _solve8(rows, [c for p in pb for c in p])


def _desk(w: int, h: int, rng: random.Random) -> Image.Image:
    """A dark, faintly speckled surface to photograph the page on.

    The desk stays dark on purpose: a desk as bright as the paper makes the
    page boundary invisible, which is not a harder test, it is an impossible
    one. Real desks are darker than real paper.
    """
    img = Image.new("RGB", (w, h), (48, 46, 44))
    px = img.load()
    for _ in range(int(w * h * 0.01)):
        x, y = rng.randrange(w), rng.randrange(h)
        v = rng.randrange(34, 72)
        px[x, y] = (v, v, v - 4)
    return img


def _photo_degrade(page: Image.Image, seed: int, downscale: float) -> Image.Image:
    """Phone-photo degradation: perspective, desk, uneven light, JPEG.

    The page quad is deliberately asymmetric (top edge shorter and higher
    than the bottom) because that is what a camera held above the paper
    actually sees, and a symmetric trapezoid is easier to correct than a
    real one -- which would flatter the preprocessing pass.
    """
    rng = random.Random(seed)
    dw, dh = 1000, 1080
    # A shadow across the PAPER, not just the desk: a hand or a lamp at the
    # edge of the frame. Without this the photo tier would only test the
    # desk background and the illumination pass would have nothing to do.
    ramp = Image.linear_gradient("L").rotate(-25).resize(page.size)
    shade = ramp.point(lambda v: int(170 + 85 * v / 255))
    page = ImageChops.multiply(
        page, Image.merge("RGB", (shade, shade, shade)))
    quad = [(150, 105), (855, 60), (905, 1015), (95, 1035)]
    coeffs = _find_coeffs(quad, [(0, 0), (page.width, 0),
                                 (page.width, page.height), (0, page.height)])
    warped = page.convert("RGBA").transform(
        (dw, dh), Image.PERSPECTIVE, coeffs, Image.BICUBIC,
        fillcolor=(0, 0, 0, 0))
    out = Image.alpha_composite(_desk(dw, dh, rng).convert("RGBA"), warped)

    # Uneven light: a rotated linear gradient over the whole frame. It never
    # lifts the desk above ~100, so the page edge stays detectable everywhere
    # (a boundary you cannot see is not a harder test, it is a broken one).
    ramp = Image.linear_gradient("L").rotate(38).resize((dw, dh))
    factor = ramp.point(lambda v: int(100 + 110 * v / 255))
    shade = Image.merge("RGB", (factor, factor, factor))
    out = ImageChops.multiply(out.convert("RGB"), shade)

    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=76)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")

    if downscale < 1.0:
        out = out.resize((int(dw * downscale), int(dh * downscale)),
                         Image.LANCZOS)
        out = out.filter(ImageFilter.GaussianBlur(1.1))
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=70)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out


def render(inv: dict, tier: str, seed: int) -> tuple[Image.Image, list[str]]:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_h1 = _font(FONTS[inv["lang"]][0], 34)
    f_h2 = _font(FONTS[inv["lang"]][0], 26)
    f_body = _font(FONTS[inv["lang"]][0], 23)
    f_small = _font(FONTS[inv["lang"]][0], 19)

    gt: list[str] = []
    y = MARGIN
    d.text((MARGIN, y), inv["vendor"], font=f_h1, fill="black"); gt.append(inv["vendor"]); y += 46
    d.text((MARGIN, y), inv["address"], font=f_small, fill="black"); gt.append(inv["address"]); y += 42
    # Punctuation follows the locale. A US invoice would not use a full-width
    # colon, and a fixture that does is not a photo of a real invoice -- it is
    # a photo of this generator. Both forms survive normalisation, so the
    # ground-truth line is built from the same pieces the page shows.
    gap = "  " if inv["lang"] == "en" else "　　"
    colon = ": " if inv["lang"] == "en" else "："
    head = f'{inv["doc_title"]}{gap}{inv["doc_no_label"]}{colon}{inv["doc_no"]}'
    d.text((MARGIN, y), head, font=f_h2, fill="black"); gt.append(head); y += 40
    datel = f'{inv["date_label"]}{colon}{inv["date"]}'
    d.text((MARGIN, y), datel, font=f_body, fill="black"); gt.append(datel); y += 44

    d.line((MARGIN, y, W - MARGIN, y), fill="black", width=2); y += 16

    header_bottom = y
    for i, h in enumerate(inv["col_headers"]):
        anchor = "la" if i == 0 else "ra"
        d.text((COL_X[i], y), h, font=f_body, fill="black", anchor=anchor)
        gt.append(h)
        header_bottom = max(
            header_bottom, d.textbbox((COL_X[i], y), h, font=f_body,
                                      anchor=anchor)[3])
    y = header_bottom + 14
    d.line((MARGIN, y, W - MARGIN, y), fill="grey", width=1)
    y += 14

    items_bottom = y
    for row in inv["items"]:
        for i, cell in enumerate(row):
            anchor = "la" if i == 0 else "ra"
            d.text((COL_X[i], y), cell, font=f_body, fill="black", anchor=anchor)
            gt.append(cell)
            items_bottom = max(
                items_bottom,
                d.textbbox((COL_X[i], y), cell, font=f_body, anchor=anchor)[3])
        y += 36

    # Rules are measured off the glyphs, not off the row grid. A rule drawn at
    # "current y minus 18" lands inside the text: every page would then print a
    # line through its own numbers, which no real invoice does, and the line
    # crosses exactly the digits this corpus exists to score.
    y = items_bottom + 14
    d.line((452, y, W - MARGIN, y), fill="grey", width=1)
    y += 18

    totals_bottom = y
    for lbl, val in (
        (inv["subtotal_label"], inv["subtotal"]),
        (inv["tax_label"], inv["tax"]),
        (inv["total_label"], inv["total"]),
    ):
        d.text((452, y), lbl, font=f_body, fill="black", anchor="ra"); gt.append(lbl)
        d.text((W - MARGIN, y), val, font=f_body, fill="black", anchor="ra"); gt.append(val)
        totals_bottom = max(
            totals_bottom,
            d.textbbox((452, y), lbl, font=f_body, anchor="ra")[3],
            d.textbbox((W - MARGIN, y), val, font=f_body, anchor="ra")[3])
        y += 36
    d.line((452, totals_bottom + 6, W - MARGIN, totals_bottom + 6),
           fill="black", width=2)

    if tier == "scan":
        rng = random.Random(seed)
        img = img.rotate(1.5, resample=Image.BICUBIC, fillcolor="white")
        img = img.filter(ImageFilter.GaussianBlur(0.7))
        px = img.load()
        for _ in range(int(W * H * 0.004)):
            x = rng.randrange(W); yy = rng.randrange(H)
            v = rng.randrange(120, 235)
            px[x, yy] = (v, v, v)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=62)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    elif tier == "photo":
        img = _photo_degrade(img, seed, downscale=1.0)
    elif tier == "hard":
        img = _photo_degrade(img, seed, downscale=0.55)
    return img, gt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["clean", "scan", "photo", "hard", "both",
                                       "all"],
                    default="all")
    ap.add_argument("--lang", default=None, help="filter to one language code")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    check_template_arithmetic()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(root, "fixtures", "cjk_synthetic")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    if args.tier == "both":      # the pre-photo pair, kept for old reports
        tiers = ["clean", "scan"]
    elif args.tier == "all":
        tiers = ["clean", "scan", "photo", "hard"]
    else:
        tiers = [args.tier]
    records = []
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        # Tier-at-a-time regeneration must not drop the other tiers: merge,
        # replacing only the records this run is about to rewrite.
        with io.open(manifest_path, encoding="utf-8") as f:
            old = json.load(f).get("records", [])
        records = [r for r in old if r["tier"] not in tiers]
    for inv in INVOICES:
        if args.lang and inv["lang"] != args.lang:
            continue
        for tier in tiers:
            # crc32, not hash(): str hash is per-process randomised, which
            # would make every regeneration produce different noise.
            seed = zlib.crc32(("%s|%s" % (inv["id"], tier)).encode("utf-8"))
            img, gt = render(inv, tier, seed)
            name = f'{inv["lang"]}__{inv["id"]}__{tier}.png'
            img.save(os.path.join(img_dir, name))
            records.append({
                "file": f"images/{name}",
                "lang": inv["lang"],
                "tier": tier,
                "currency": inv["currency"],
                "font": FONTS[inv["lang"]][2],
                "lines": gt,
                "fields": {
                    "vendor": inv["vendor"],
                    "doc_no": inv["doc_no"],
                    "date": inv["date"],
                    "subtotal": inv["subtotal"],
                    "tax": inv["tax"],
                    "total": inv["total"],
                },
            })

    manifest = os.path.join(out_dir, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"count": len(records), "records": records}, f,
                  ensure_ascii=False, indent=1)
    print(f"wrote {len(records)} images -> {img_dir}")
    print(f"manifest -> {manifest}")
    by_lang = {}
    for r in records:
        by_lang.setdefault(r["lang"], 0)
        by_lang[r["lang"]] += 1
    print("per language:", by_lang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
