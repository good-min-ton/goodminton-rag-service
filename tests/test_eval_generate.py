from eval.generate_golden import (
    strip_accents,
    leaks_name,
    stratified_sample,
    parse_name_brand,
)


def test_strip_accents():
    assert strip_accents("Vợt Cầu Lông") == "vot cau long"


def test_leaks_name_detects_brand_token():
    assert leaks_name("Astrox 88D giá bao nhiêu", "Yonex Astrox 88D", "Yonex") is True
    assert leaks_name("vợt nào nhẹ cho người mới", "Yonex Astrox 88D", "Yonex") is False


def test_stratified_sample_balances_categories():
    rows = [
        {"source_id": str(i), "category": "Vợt cầu lông", "name": f"v{i}", "brand": "b"}
        for i in range(5)
    ] + [
        {
            "source_id": str(i),
            "category": "Giày cầu lông",
            "name": f"g{i}",
            "brand": "b",
        }
        for i in range(5, 8)
    ]
    picked = stratified_sample(rows, per_category=2)
    counts = {}
    for r in picked:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    assert counts == {"Vợt cầu lông": 2, "Giày cầu lông": 2}


def test_parse_name_brand():
    content = (
        "Sản phẩm: Yonex Astrox 88D\n"
        "Thương hiệu: Yonex\n"
        "Danh mục: Vợt cầu lông\n"
        "Thông số: nặng 3U\n"
        "Mô tả: vợt công thủ toàn diện"
    )
    assert parse_name_brand(content) == ("Yonex Astrox 88D", "Yonex")
    # Missing lines fall back to "".
    assert parse_name_brand("Danh mục: Vợt cầu lông") == ("", "")
