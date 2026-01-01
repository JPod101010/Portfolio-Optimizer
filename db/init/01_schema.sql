CREATE TABLE prices (
    symbol VARCHAR(10) NOT NULL,
    date_ DATE NOT NULL,
    open_price NUMERIC(14, 4),
    close_price NUMERIC(14, 4),
    high_price NUMERIC(14, 4),
    low_price NUMERIC(14, 4),
    raw_diff NUMERIC(14, 4),
    percent_diff NUMERIC(14, 8),
    logpercent_diff NUMERIC(14, 8),
    volume BIGINT,
    PRIMARY KEY (symbol, date_)
);

-- Indexing for performance
CREATE INDEX idx_prices_symbol ON prices(symbol);
CREATE INDEX idx_prices_date ON prices(date_);