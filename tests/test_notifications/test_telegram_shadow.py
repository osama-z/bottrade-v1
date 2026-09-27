"""Task 5.1 Telegram — shadow-mode alerts are clearly marked, not executions."""
from notifications.telegram_bot import TelegramBot


def _bot():
    bot = TelegramBot.__new__(TelegramBot)     # bypass __init__ (no token needed)
    bot.sent = []
    bot._send = bot.sent.append                # capture the message
    return bot


class TestShadowSignalMessage:
    def _msg(self, **over):
        bot = _bot()
        kw = dict(pair="BTC/USDT", decision="BUY", price=100.0,
                  expected_slippage=0.0012, expected_fill=100.12,
                  confidence=0.9, reason="filters passed")
        kw.update(over)
        bot.send_shadow_signal(**kw)
        return bot.sent[0]

    def test_marked_shadow_not_executed(self):
        msg = self._msg()
        assert "SHADOW SIGNAL" in msg and "no order placed" in msg
        assert "TRADE EXECUTED" not in msg
        assert "New Trade Opened" not in msg          # not the real-fill message

    def test_includes_signal_price_expected_fill_and_slippage(self):
        msg = self._msg(price=100.0, expected_fill=100.12, expected_slippage=0.0012)
        assert "$100.00" in msg                       # signal price
        assert "$100.12" in msg                        # expected fill (from CostModel)
        assert "0.1200%" in msg                        # modelled slippage

    def test_confirms_logged_to_shadow_decisions(self):
        assert "shadow_decisions" in self._msg()

    def test_confidence_and_direction(self):
        buy = self._msg(decision="BUY", confidence=0.9)
        assert "90%" in buy and "📈 BUY" in buy
        sell = self._msg(decision="SELL", expected_fill=99.9)
        assert "📉 SELL" in sell

    def test_reason_optional(self):
        msg = self._msg(reason="")
        assert "SHADOW SIGNAL" in msg                  # still valid with no reason


class TestSystemStartMode:
    def test_shadow_mode_is_shown_not_paper(self):
        bot = _bot()
        bot.send_system_start(balance=10_000.0, pairs=["BTC/USDT"],
                              mode="🔍 SHADOW (log-only, no orders)")
        msg = bot.sent[0]
        assert "SHADOW" in msg
        assert "Mode: `PAPER TRADING`" not in msg      # the old hardcoded label is gone

    def test_mode_is_parameterized(self):
        bot = _bot()
        bot.send_system_start(balance=10_000.0, pairs=["BTC/USDT"], mode="⚠️ LIVE TRADING")
        assert "LIVE TRADING" in bot.sent[0]
