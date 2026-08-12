"""Phần thuần logic của bộ sinh golden v2.

Ba generator theo luật và bộ chèn lỗi gõ không đụng tới DB hay LLM, nên kiểm
được trọn vẹn ở đây. Phần cần corpus thật (LLM pre-label, truy hồi) chỉ chạy
được trên máy chủ, nên ở đây chỉ kiểm bộ đọc phán quyết của LLM.
"""

from eval.catalog import Product, format_price_vn, parse_specs, price_thresholds
from eval.generate import (
    _doc_phan_quyet,
    doc_target,
    gen_browse,
    gen_multi_category,
    gen_price,
    gen_typo,
)
from eval.golden import QUERY_TYPES, SOURCES
from eval.typo import bo_dau, inject

VOT = "Vợt cầu lông"
GIAY = "Giày cầu lông"


def _catalog() -> list[Product]:
    ra = []
    for i in range(1, 13):
        ra.append(
            Product(
                str(i),
                f"Vợt {i}",
                "Yonex",
                VOT,
                {"Trọng lượng": "4U"},
                price=400_000 * i,
            )
        )
    for i in range(13, 21):
        ra.append(
            Product(
                str(i),
                f"Giày {i}",
                "Lining",
                GIAY,
                {"Đế": "cao su"},
                price=300_000 * (i - 12),
            )
        )
    return ra


# ------------------------------------------------------------------ catalog


def test_parse_specs_doc_dong_thong_so():
    content = (
        "Sản phẩm: Yonex Astrox 88D\nThương hiệu: Yonex\nDanh mục: Vợt cầu lông\n"
        "Thông số: Trọng lượng: 3U | Độ căng: 24-28 lbs\nMô tả: abc"
    )
    assert parse_specs(content) == {"Trọng lượng": "3U", "Độ căng": "24-28 lbs"}


def test_parse_specs_khong_khai_thi_rong():
    assert parse_specs("Thông số: N/A") == {}
    assert parse_specs("Danh mục: Vợt cầu lông") == {}


def test_format_price_vn():
    assert format_price_vn(500_000) == "500k"
    assert format_price_vn(1_000_000) == "1 triệu"
    assert format_price_vn(1_500_000) == "1,5 triệu"


def test_price_thresholds_nam_trong_dai_gia_that():
    muc = price_thresholds(_catalog(), VOT)
    gia = [p.price for p in _catalog() if p.category == VOT]
    assert muc, "phải có ít nhất một ngưỡng"
    for m in muc:
        # Ngưỡng vô dụng là ngưỡng không chia được tập: rỗng hoặc lấy tất.
        duoi = [g for g in gia if g <= m]
        assert 0 < len(duoi) < len(gia)


def test_price_thresholds_khong_co_gia_thi_rong():
    khong_gia = [Product("1", "x", "y", VOT, {}, price=None)]
    assert price_thresholds(khong_gia, VOT) == []


# ------------------------------------------------------------------ typo


def test_bo_dau():
    assert bo_dau("Vợt cầu lông đỏ") == "Vot cau long do"


def test_inject_tat_dinh_va_khac_cau_goc():
    q = "vợt cầu lông nhẹ cho người mới"
    for seed in range(20):
        a, b = inject(q, seed), inject(q, seed)
        assert a == b, "cùng seed phải ra cùng kết quả"
        assert a != q, f"seed {seed} không chèn được lỗi nào"


def test_inject_phu_cac_kieu_loi():
    q = "vợt cầu lông nhẹ giá rẻ"
    ra = {inject(q, s) for s in range(8)}
    assert len(ra) >= 3, "bốn kiểu lỗi phải cho ra nhiều biến thể khác nhau"


# ------------------------------------------------------------------ generator


def test_gen_browse_du_so_va_nhan_dung_danh_muc():
    ds = gen_browse(_catalog(), 10)
    assert len(ds) == 10
    for d in ds:
        assert d["query_type"] == "browse"
        assert d["source"] == "auto"
        cat = d["expected_categories"][0]
        mong_doi = {p.source_id for p in _catalog() if p.category == cat}
        assert set(d["relevant_source_ids"]) == mong_doi


def test_gen_browse_khong_lap_cau_hoi():
    ds = gen_browse(_catalog(), 12)
    assert len({d["query"] for d in ds}) == len(ds)


def test_gen_multi_category_nhan_la_hop_hai_danh_muc():
    ds = gen_multi_category(_catalog(), 4)
    assert ds
    for d in ds:
        a, b = d["expected_categories"]
        mong_doi = {p.source_id for p in _catalog() if p.category in (a, b)}
        assert set(d["relevant_source_ids"]) == mong_doi


def test_gen_price_nhan_dung_tran_gia():
    ds = gen_price(_catalog(), 6)
    assert ds
    tra_cuu = {p.source_id: p for p in _catalog()}
    for d in ds:
        assert d["price_constrained"] is True
        assert d["query_type"] == "price-constrained"
        muc = d["provenance"]["price_threshold"]
        cat = d["expected_categories"][0]
        for sid in d["relevant_source_ids"]:
            assert tra_cuu[sid].category == cat
            assert tra_cuu[sid].price <= muc


def test_gen_price_bo_qua_san_pham_chua_co_gia():
    cat = _catalog()
    cat[0] = Product(cat[0].source_id, cat[0].name, cat[0].brand, VOT, {}, price=None)
    for d in gen_price(cat, 5):
        assert cat[0].source_id not in d["relevant_source_ids"]


def test_gen_typo_ke_thua_nhan_cua_cau_goc():
    nguon = gen_browse(_catalog(), 6) + gen_multi_category(_catalog(), 4)
    theo_id = {d["id"]: d for d in nguon}
    ds = gen_typo(nguon, 8)
    assert ds
    for d in ds:
        goc = theo_id[d["provenance"]["derived_from"]]
        assert d["relevant_source_ids"] == goc["relevant_source_ids"]
        assert d["expected_categories"] == goc["expected_categories"]
        assert d["query"] != goc["query"]


def test_gen_typo_rai_deu_tren_cac_loai_nguon():
    nguon = gen_browse(_catalog(), 8) + gen_multi_category(_catalog(), 8)
    ds = gen_typo(nguon, 8)
    loai = {d["provenance"]["derived_from"].split("-")[1] for d in ds}
    assert len(loai) >= 2, "không được dồn hết vào một loại nguồn"


# ------------------------------------------------------------------ khác


def test_moi_dong_hop_le_voi_schema_loader():
    ds = (
        gen_browse(_catalog(), 3)
        + gen_multi_category(_catalog(), 3)
        + gen_price(_catalog(), 3)
    )
    for d in ds:
        assert d["query_type"] in QUERY_TYPES
        assert d["source"] in SOURCES
        assert d["relevant_source_ids"], "loader từ chối dòng có nhãn rỗng"
        assert d["provenance"]["generator_version"] == "v2"


def test_doc_phan_quyet():
    raw = "1. CO\n2. KHONG\n3. CO\n4. gì đó\n99. CO"
    assert _doc_phan_quyet(raw, 4) == {1, 3}


def test_doc_target_doi_ten_price():
    assert doc_target("attribute=40,price=25") == {
        "attribute": 40,
        "price-constrained": 25,
    }
