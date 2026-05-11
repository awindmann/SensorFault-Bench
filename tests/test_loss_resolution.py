from metrics.loss import BASE_LOSS_NAMES, resolve_stateless_loss


def test_resolve_stateless_loss_supports_maintained_losses():
    for loss_name in BASE_LOSS_NAMES:
        loss_fn = resolve_stateless_loss(loss_name)
        assert callable(loss_fn)


def test_resolve_stateless_loss_rejects_unknown_losses():
    try:
        resolve_stateless_loss("not_a_loss")
    except KeyError as exc:
        assert "not_a_loss" in str(exc)
    else:
        raise AssertionError("Unknown loss should raise.")
