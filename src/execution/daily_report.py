# -------------------------
# file: daily_report.py
# -------------------------
"""
Generate daily PnL reports (CSV + DB summary) and optionally send/email reports.
"""
import csv
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_models import TradeLog

TRADE_DB_URL = "postgresql+psycopg2://trader:trader123@localhost:5432/trades"
engine = create_engine(TRADE_DB_URL)
Session = sessionmaker(bind=engine)


def generate_daily_report(for_date=None, out_path=None):
    d = for_date or datetime.date.today()
    start = datetime.datetime.combine(d, datetime.time.min)
    end = datetime.datetime.combine(d, datetime.time.max)
    session = Session()
    rows = session.query(TradeLog).filter(TradeLog.closed_at >= start, TradeLog.closed_at <= end).all()
    total_realised = sum((r.realised or 0.0) for r in rows)
    total_unreal = sum((r.unreal or 0.0) for r in session.query(TradeLog).filter(TradeLog.status=='open').all())
    report = {
        'date': d.isoformat(),
        'realised': total_realised,
        'unreal': total_unreal,
        'closed_trades': len(rows)
    }
    # write CSV
    out_path = out_path or f"daily_report_{d.isoformat()}.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['plan_id','realised','closed_at','meta'])
        for r in rows:
            writer.writerow([r.plan_id, r.realised, r.closed_at.isoformat() if r.closed_at else '', str(r.meta)])
    session.close()
    return report

if __name__ == '__main__':
    print(generate_daily_report())