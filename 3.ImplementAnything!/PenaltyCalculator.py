from datetime import datetime

class PenaltyCalculator:
    def __init__(self, monthly_fee = 350000, monthly_dues = 180000 ,deposit = 800000, expiration_date = '2021/06/08'):
        """
        Inputs:
            월세, 공과금, 보증금, 최초계약만료일
        """
        self.monthly_fee = monthly_fee
        self.monthly_dues = monthly_dues
        self.deposit = deposit
        self.expiration_date = self._to_datetime(expiration_date)
        self.start = self.expiration_date.day

    def _to_datetime(self, date):
        return datetime(*map(int, date.split('/')))

    def deposit_penalty(self,notice_date_time,exit_date_time):
        """
        고지시점부터 퇴실희망일까지 남은 기간에 따라 보증금의 일부를 위약금으로 부여한다.
        """
        remaining_days = (exit_date_time - notice_date_time).days

        if 1 <= remaining_days < 7:
            return remaining_days, self.deposit * 0.50
        elif 7 <= remaining_days < 15:
            return remaining_days, self.deposit * 0.42
        elif 15 <= remaining_days < 30:
            return remaining_days, self.deposit * 0.34
        elif 30 <= remaining_days < 45:
            return remaining_days, self.deposit * 0.26
        elif 45 <= remaining_days < 60:
            return remaining_days, self.deposit * 0.18
        else:
            return remaining_days, self.deposit * 0.10

    def expiration_penalty(self, exit_date_time):
        """
        퇴실희망일부터 계약만료일까지 남은 기간에 따라 보증금의 일부를 위약금으로 부여한다.
        """

        remaining_days = (self.expiration_date - exit_date_time).days

        return self.deposit * 0.005 * remaining_days

    def get_monthly_fee(self, exit_date_time, next_start_time):
        """
        n일간의 추가 월세를 계산한다.
        """
        remaining_days = (exit_date_time - next_start_time).days

        if remaining_days <= 0:
            return 0,0

        return remaining_days, remaining_days * self.monthly_fee / 30 + 30000 # 관리비 3만원!

    def get_monthly_dues(self, exit_date_time, next_start_time, share = 4):
        """
        공과금을 일할 계산한다. ( 4명이 나눠냄 ! )
        """
        remaining_days = (exit_date_time - next_start_time).days

        if remaining_days <= 0:
            return 0

        return remaining_days * self.monthly_dues/(30*(share - 1) + remaining_days)

    def get_penalty(self, notice_date , exit_date, share = 4):
        """
        공지 일자와 퇴실 희망일을 이용하여 총 위약금을 계산한다.
        """
        notice_date_time = self._to_datetime(notice_date)
        exit_date_time = self._to_datetime(exit_date)

        # 1. 광고비
        advertising = 50000

        # 2. 고지시점에 따른 부과액
        n_day, notice = self.deposit_penalty(notice_date_time, exit_date_time)

        # 3. 잔여일수에 따른 부과액
        remaining_days = self.expiration_penalty(exit_date_time)

        # 다음달 계약 시작일
        next_start = datetime(self.expiration_date.year, notice_date_time.month+1, self.start)

        # 4. n일치 추가 월세
        r_day, fee = self.get_monthly_fee(exit_date_time, next_start)

        # 6. n일치 추가 공과금
        dues = self.get_monthly_dues(exit_date_time, next_start, share = share)

        print(f"""
        --------------------------------------------------
        {notice_date}에 공지하여 {exit_date}까지 입주하는 경우, 
        --------------------------------------------------
        > 광고비\t\t\t\t\t\t\t: {advertising:.3f} 
        > 고지 시점({n_day}일 전)에 따른 부과액\t: {notice:.3f}
        > 잔여 일수에 따른 부과액\t\t\t: {remaining_days:.3f}
        > {r_day}일간 추가 월세 및 관리비\t\t\t: {fee:.3f}
        > {r_day}일간 추가 공과금\t\t\t\t: {dues:.3f}
        --------------------------------------------------
        >>> total : {advertising + notice + remaining_days + fee + dues:.3f} = {advertising + notice + remaining_days:.3f} + {fee + dues:.3f}
        --------------------------------------------------
        """)

if __name__ == '__main__':
    cal = PenaltyCalculator(monthly_fee = 350000, monthly_dues = 180000 ,deposit = 800000, expiration_date = '2021/06/08')
    cal.get_penalty(notice_date = '2021/02/27', exit_date = '2021/03/08')
    cal.get_penalty(notice_date = '2021/02/27', exit_date = '2021/03/14')
    cal.get_penalty(notice_date = '2021/02/27', exit_date = '2021/03/29')
