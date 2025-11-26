# -------------------------
# file: db_models.py
# -------------------------
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class TradeLog(Base):
    __tablename__ = 'trade_logs'
    id = Column(Integer, primary_key=True)
    plan_id = Column(String, index=True)
    side = Column(String)
    short_strike = Column(Integer)
    long_strike = Column(Integer)
    credit = Column(Float)
    qty = Column(Integer)
    opened_at = Column(DateTime, default=datetime.datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    status = Column(String, default='open')
    realised = Column(Float, default=0.0)
    unreal = Column(Float, default=0.0)
    meta = Column(JSON)

class MarketFeature(Base):
    __tablename__ = 'market_features'
    id = Column(Integer, primary_key=True)
    feature = Column(String)
    value = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)