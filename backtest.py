# backtest.py

import pandas as pd

def simulate_strict_investor(
	test_dates,
	y_test,
	y_pred,
	initial_balance=10_000_000,
	fee_rate=0.05,
	max_inventory=5,
	target_margin=0.10,
):
	balance = initial_balance
	item_count = 0
	avg_buy_price = 0.0

	trades = []

	# Series/ndarray → 1차원으로 깔끔하게 맞춰두기
	test_dates = pd.Series(test_dates).reset_index(drop=True)
	y_test = pd.Series(y_test).reset_index(drop=True)
	y_pred = pd.Series(y_pred).reset_index(drop=True)

	for date, real_price, pred_price in zip(test_dates, y_test, y_pred):
		real_price = float(real_price)
		pred_price = float(pred_price)

		# 매수 조건
		if balance >= real_price and item_count < max_inventory:
			expected_profit_margin = (pred_price - real_price) / real_price

			if expected_profit_margin > target_margin:
				balance -= real_price
				item_count += 1

				if item_count == 1:
					avg_buy_price = real_price
				else:
					avg_buy_price = (
						avg_buy_price * (item_count - 1) + real_price
					) / item_count

				trades.append({
					"type": "BUY",
					"date": date,
					"price": real_price,
					"pred_price": pred_price,
					"expected_margin": expected_profit_margin,
					"profit": None,
				})

		# 매도 조건
		elif item_count > 0:
			current_profit_rate = (real_price - avg_buy_price) / avg_buy_price

			if real_price >= pred_price or current_profit_rate > 0.05:
				sell_amount = real_price * (1 - fee_rate)
				profit = sell_amount - avg_buy_price
				balance += sell_amount
				item_count -= 1

				trades.append({
					"type": "SELL",
					"date": date,
					"price": real_price,
					"pred_price": pred_price,
					"expected_margin": None,
					"profit": profit,
				})

				if item_count == 0:
					avg_buy_price = 0

	# 남은 물량 평가
	last_price = float(y_test.iloc[-1])
	final_asset_value = balance + (item_count * last_price * (1 - fee_rate))
	net_profit = final_asset_value - initial_balance
	roi = (net_profit / initial_balance) * 100

	trade_df = pd.DataFrame(trades)

	return {
		"final_asset_value": float(final_asset_value),
		"net_profit": float(net_profit),
		"roi": float(roi),
		"trade_history": trade_df,
	}

