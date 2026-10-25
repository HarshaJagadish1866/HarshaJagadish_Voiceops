#!/usr/bin/env python3
"""ZentroMart Database Loader & Business Day Calculator.

This script loads the 7 CSV files for ZentroMart using pd.read_csv(..., dtype=str),
maps their relational structure, defines the anchor date (20 October 2026), and
implements business day calculation logic that skips Sundays and warehouse holidays.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Anchor "Today" for ZentroMart Customer Support Agent
TODAY: pd.Timestamp = pd.Timestamp("2026-10-20")
TODAY_DATE: date = TODAY.date()

# Standard working week: Monday (0) to Saturday (5) are business days; Sunday (6) is non-working.
# NumPy 7-character weekmask string (Mon-Sun): '1111110'
WEEK_MASK: str = "1111110"


def load_raw_tables(data_dir: Union[str, Path] = "data") -> Dict[str, pd.DataFrame]:
    """Read the 7 CSV files from data/ directory using pd.read_csv(..., dtype=str).

    Parameters:
        data_dir: Path to the data directory containing CSV files.

    Returns:
        Dictionary mapping table name to raw DataFrame.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        for fallback in [
            Path.cwd() / "data",
            Path.cwd() / "suppose delete" / "data",
            Path.cwd().parent / "data",
            Path.cwd().parent / "suppose delete" / "data",
            Path(__file__).resolve().parent.parent / "data",
            Path(__file__).resolve().parent.parent / "suppose delete" / "data",
        ]:
            if fallback.exists():
                data_path = fallback
                break
    files = [
        "customers.csv",
        "orders.csv",
        "order_items.csv",
        "products.csv",
        "pincodes.csv",
        "holidays.csv",
        "support_logs.csv",
    ]

    tables: Dict[str, pd.DataFrame] = {}
    for filename in files:
        filepath = data_path / filename
        table_key = Path(filename).stem
        if not filepath.exists():
            raise FileNotFoundError(f"Missing required CSV file: {filepath}")
        tables[table_key] = pd.read_csv(filepath, dtype=str)

    return tables


class BusinessDayCalculator:
    """Calculates business days according to ZentroMart policy:

    - Sundays are non-working days.
    - Monday through Saturday are working days.
    - Dates listed in holidays.csv are excluded.
    """

    def __init__(self, holidays_df: pd.DataFrame):
        self.holidays_df = holidays_df
        # Parse holiday dates
        raw_dates = holidays_df["date"].dropna().str.strip()
        self.holiday_dates: Set[date] = {
            pd.to_datetime(d).date() for d in raw_dates if d
        }
        self.holiday_dates_np = [
            np.datetime64(d.isoformat(), "D") for d in sorted(self.holiday_dates)
        ]

    def is_business_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> bool:
        """Check if a date is a business day (Mon-Sat, not in holidays)."""
        if isinstance(d, str):
            dt = pd.to_datetime(d).date()
        elif isinstance(d, (datetime, pd.Timestamp)):
            dt = d.date()
        else:
            dt = d

        # Sunday is 6
        if dt.weekday() == 6:
            return False
        if dt in self.holiday_dates:
            return False
        return True

    def count_business_days(
        self,
        start_date: Union[str, date, datetime, pd.Timestamp],
        end_date: Union[str, date, datetime, pd.Timestamp],
    ) -> int:
        """Count business days between start_date and end_date (half-open [start, end)).

        Uses NumPy's vectorized busday_count with the Mon-Sat weekmask.
        """
        s = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        e = pd.to_datetime(end_date).strftime("%Y-%m-%d")

        s_np = np.datetime64(s, "D")
        e_np = np.datetime64(e, "D")

        return int(
            np.busday_count(
                s_np,
                e_np,
                weekmask=WEEK_MASK,
                holidays=self.holiday_dates_np,
            )
        )

    def add_business_days(
        self,
        start_date: Union[str, date, datetime, pd.Timestamp],
        days: int,
        roll: str = "forward",
    ) -> date:
        """Add (or subtract) n business days to start_date, skipping Sundays & holidays.

        Parameters:
            start_date: Base date.
            days: Number of business days to offset.
            roll: How to treat non-business start dates ('forward', 'backward', 'raise').
        """
        s = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        s_np = np.datetime64(s, "D")

        offset_np = np.busday_offset(
            s_np,
            days,
            roll=roll,
            weekmask=WEEK_MASK,
            holidays=self.holiday_dates_np,
        )
        return pd.to_datetime(str(offset_np)).date()

    def business_days_since(
        self, past_date: Union[str, date, datetime, pd.Timestamp]
    ) -> int:
        """Count business days elapsed from past_date to TODAY (20 Oct 2026)."""
        return self.count_business_days(past_date, TODAY)

    def business_days_until(
        self, future_date: Union[str, date, datetime, pd.Timestamp]
    ) -> int:
        """Count business days remaining from TODAY (20 Oct 2026) to future_date."""
        return self.count_business_days(TODAY, future_date)


class ZentroMartDatabase:
    """Manages the relational data and business calendar for ZentroMart."""

    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.tables: Dict[str, pd.DataFrame] = load_raw_tables(self.data_dir)

        # Raw DataFrames
        self.customers: pd.DataFrame = self.tables["customers"]
        self.orders: pd.DataFrame = self.tables["orders"]
        self.order_items: pd.DataFrame = self.tables["order_items"]
        self.products: pd.DataFrame = self.tables["products"]
        self.pincodes: pd.DataFrame = self.tables["pincodes"]
        self.holidays: pd.DataFrame = self.tables["holidays"]
        self.support_logs: pd.DataFrame = self.tables["support_logs"]

        # Today's reference date
        self.today: pd.Timestamp = TODAY
        self.today_date: date = TODAY_DATE

        # Business day calculator
        self.calendar: BusinessDayCalculator = BusinessDayCalculator(self.holidays)

        # Relational joins
        self.orders_with_customers = self._join_orders_customers()
        self.items_with_orders = self._join_items_orders()
        self.items_with_products = self._join_items_products()
        self.orders_with_pincodes = self._join_orders_pincodes()
        self.master_order_items = self._build_master_view()

    def _join_orders_customers(self) -> pd.DataFrame:
        """Join orders to customers using customer_id."""
        return self.orders.merge(
            self.customers,
            on="customer_id",
            how="left",
            suffixes=("", "_customer"),
        )

    def _join_items_orders(self) -> pd.DataFrame:
        """Join order_items to orders using order_id."""
        return self.order_items.merge(
            self.orders,
            on="order_id",
            how="left",
            suffixes=("", "_order"),
        )

    def _join_items_products(self) -> pd.DataFrame:
        """Join order_items to products using sku."""
        return self.order_items.merge(
            self.products,
            on="sku",
            how="left",
            suffixes=("", "_catalog"),
        )

    def _join_orders_pincodes(self) -> pd.DataFrame:
        """Join orders.delivery_pincode to pincodes.pincode."""
        return self.orders.merge(
            self.pincodes,
            left_on="delivery_pincode",
            right_on="pincode",
            how="left",
            suffixes=("", "_pincode"),
        )

    def _build_master_view(self) -> pd.DataFrame:
        """Comprehensive consolidated view:

        order_items -> products (on sku) -> orders (on order_id) -> customers (on customer_id)
        and delivery pincode info (on delivery_pincode -> pincode).
        """
        # 1. items + products
        df = self.order_items.merge(
            self.products,
            on="sku",
            how="left",
            suffixes=("", "_catalog"),
        )
        # 2. + orders
        df = df.merge(
            self.orders,
            on="order_id",
            how="left",
            suffixes=("", "_order"),
        )
        # 3. + customers
        df = df.merge(
            self.customers,
            on="customer_id",
            how="left",
            suffixes=("", "_customer"),
        )
        # 4. + pincodes
        df = df.merge(
            self.pincodes,
            left_on="delivery_pincode",
            right_on="pincode",
            how="left",
            suffixes=("", "_pincode"),
        )
        return df


def main():
    print(f"=== ZentroMart Database Loader ===")
    print(f"Today's simulated date: {TODAY.strftime('%d %B %Y')} ({TODAY_DATE})")

    # Locate data directory relative to this script
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"

    db = ZentroMartDatabase(data_dir=data_dir)

    print("\n--- 1. Loaded 7 CSV Tables (all dtype=str) ---")
    for name, df in db.tables.items():
        print(f"  • {name:15s}: {df.shape[0]:>5} rows, {df.shape[1]:>3} columns")

    print("\n--- 2. Relational Joins ---")
    print(f"  • orders + customers (customer_id)               : {db.orders_with_customers.shape}")
    print(f"  • order_items + orders (order_id)                : {db.items_with_orders.shape}")
    print(f"  • order_items + products (sku)                   : {db.items_products_shape if hasattr(db, 'items_products_shape') else db.items_with_products.shape}")
    print(f"  • orders.delivery_pincode + pincodes.pincode     : {db.orders_with_pincodes.shape}")
    print(f"  • Master view (items + orders + cust + pin)      : {db.master_order_items.shape}")

    print("\n--- 3. Business Day Calculations (Mon-Sat, skipping Sundays & Holidays) ---")
    print(f"Holidays loaded from data/holidays.csv:")
    for _, row in db.holidays.iterrows():
        print(f"  • {row['date']}: {row['name']}")

    # Example: Refund initiated on 17 October 2026 (Saturday)
    # Card refund timeline: 5 to 7 business days
    sample_refund_date = "2026-10-17"
    exp_5 = db.calendar.add_business_days(sample_refund_date, 5)
    exp_7 = db.calendar.add_business_days(sample_refund_date, 7)
    days_to_today = db.calendar.business_days_since(sample_refund_date)

    print(f"\nExample Scenario (Refund initiated on {sample_refund_date}):")
    print(f"  • Business days elapsed until today ({TODAY_DATE}): {days_to_today} business day(s)")
    print(f"  • 5 business days deadline (min card refund window): {exp_5}")
    print(f"  • 7 business days deadline (max card refund window): {exp_7}")


if __name__ == "__main__":
    main()
