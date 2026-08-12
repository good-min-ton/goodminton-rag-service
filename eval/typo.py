"""Chèn lỗi gõ tiếng Việt vào một câu hỏi đúng chính tả.

Hoàn toàn tất định (theo seed) và không dùng LLM: slice `typo` chỉ đo khả năng
chịu lỗi gõ của truy hồi, nên câu hỏi phải tái tạo được y hệt ở lần chạy sau,
nếu không thì so sánh trước/sau fine-tune mất ý nghĩa.

Bốn kiểu lỗi mô phỏng cách gõ thật, không phải nhiễu ngẫu nhiên trên ký tự:

- mất dấu      người gõ nhanh bỏ dấu: "vợt" -> "vot"
- telex sót    gõ telex nhưng thiếu bước bỏ dấu: "nhẹ" -> "nhej"
- phím kề      trượt tay sang phím bên cạnh trên QWERTY
- viết tắt     lối viết tắt phổ biến khi chat: "cầu lông" -> "cl"

Nhãn của câu gốc được kế thừa nguyên vẹn, vì lỗi gõ không làm đổi ý định.
"""

import random
import unicodedata

# Ký tự telex đứng sau âm tiết để tạo dấu. Người gõ vội hay để lộ chính chữ này.
TELEX_DAU = {
    "̀": "f",  # huyền
    "́": "s",  # sắc
    "̃": "x",  # ngã
    "̉": "r",  # hỏi
    "̣": "j",  # nặng
}

VIET_TAT = {
    "cầu lông": "cl",
    "không": "ko",
    "được": "dc",
    "khoảng": "khoang",
    "người mới": "ng moi",
    "sản phẩm": "sp",
    "giá rẻ": "gia re",
}

# Các phím kề nhau trên QWERTY, chỉ lấy chữ thường không dấu.
PHIM_KE = {
    "a": "sq",
    "b": "vn",
    "c": "xv",
    "d": "sf",
    "e": "wr",
    "f": "dg",
    "g": "fh",
    "h": "gj",
    "i": "uo",
    "j": "hk",
    "k": "jl",
    "l": "k",
    "m": "n",
    "n": "bm",
    "o": "ip",
    "p": "o",
    "q": "wa",
    "r": "et",
    "s": "ad",
    "t": "ry",
    "u": "yi",
    "v": "cb",
    "w": "qe",
    "x": "zc",
    "y": "tu",
    "z": "x",
}


def bo_dau(s: str) -> str:
    """Bỏ toàn bộ dấu thanh và dấu mũ, giữ nguyên chữ cái."""
    nfd = unicodedata.normalize("NFD", s)
    khong_dau = "".join(c for c in nfd if not unicodedata.combining(c))
    return khong_dau.replace("đ", "d").replace("Đ", "D")


def _telex_sot(tu: str) -> str:
    """Bỏ dấu nhưng để lại ký tự telex tương ứng ở cuối âm tiết."""
    nfd = unicodedata.normalize("NFD", tu)
    dau = next((c for c in nfd if c in TELEX_DAU), None)
    if dau is None:
        return ""
    return bo_dau(tu) + TELEX_DAU[dau]


def _phim_ke(tu: str, rng: random.Random) -> str:
    """Đổi một chữ cái sang phím bên cạnh. Chỉ đụng chữ không dấu để lỗi trông
    giống trượt tay chứ không giống gõ sai bộ gõ."""
    vi_tri = [i for i, c in enumerate(tu) if c in PHIM_KE]
    if not vi_tri:
        return ""
    i = rng.choice(vi_tri)
    return tu[:i] + rng.choice(PHIM_KE[tu[i]]) + tu[i + 1 :]


def inject(query: str, seed: int) -> str:
    """Trả về câu đã chèn lỗi. Luôn khác câu gốc, hoặc trả lại nguyên câu nếu
    không có lỗi nào áp dụng được (câu quá ngắn, không dấu, không từ viết tắt)."""
    rng = random.Random(seed)
    kieu = seed % 4

    if kieu == 0:
        ra = bo_dau(query)
        if ra != query:
            return ra

    if kieu == 1:
        tu = query.split()
        co_dau = [i for i, t in enumerate(tu) if _telex_sot(t)]
        if co_dau:
            i = rng.choice(co_dau)
            tu[i] = _telex_sot(tu[i])
            return " ".join(tu)

    if kieu == 2:
        thap = query.lower()
        hop = [k for k in VIET_TAT if k in thap]
        if hop:
            k = rng.choice(sorted(hop))
            return thap.replace(k, VIET_TAT[k], 1)

    # kieu == 3, và cũng là đường lui cho ba kiểu trên khi không áp dụng được.
    tu = bo_dau(query).split()
    dai = [i for i, t in enumerate(tu) if len(t) >= 3]
    if dai:
        i = rng.choice(dai)
        moi = _phim_ke(tu[i], rng)
        if moi:
            tu[i] = moi
            return " ".join(tu)

    ra = bo_dau(query)
    return ra if ra != query else query
