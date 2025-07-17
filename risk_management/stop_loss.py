def apply_stop_loss(df, stop_loss_pct = 0.02):
    equity = [100_000]
    position = 0
    entry_price = None

    for i in range(1, len(df)):
        if df['position'].iloc[i] != position:
            position = df['position'].iloc[i]
            entry_price = df['price'].iloc[i]

        price = df['price'].iloc[i]
        if position != 0 and entry_price:
            pnl = (price - entry_price) / entry_price * position
            if pnl < -stop_loss_pct:
                position = 0;
                entry_price = None
        equity.append(equity[-1] * (1 + df['returns'].iloc[i] * position))

    df['equity_curve'] = equity
    return df