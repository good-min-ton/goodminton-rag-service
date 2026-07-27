def test_reuses_visible_product_id_query():
    # The image backfill must REUSE the products.is_visible id source, not roll its own.
    from scripts import backfill_product_images, backfill_products

    assert (
        backfill_product_images.fetch_product_ids is backfill_products.fetch_product_ids
    )
