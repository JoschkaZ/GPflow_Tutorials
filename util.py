


def combine_account_histories(use_fxcm = False):
    print('cominbing account histories...')
    log = []

    data_path = os.path.dirname(os.path.abspath(__file__)) + r'\\data'
    cmc_path = data_path+r'\\history_cmc.csv'
    if use_fxcm == True: fxcm_path = data_path+r'\\history_fxcm.csv'


    with open(cmc_path) as f:
        reader = csv.reader(f)
        cmc_data = list(reader)
    if use_fxcm == True:
        with open(fxcm_path) as f:
            reader = csv.reader(f)
            fxcm_data = list(reader)

    processed_cmc_data = []
    processed_fxcm_data = []
    # process cmc data
    if True:
        log.append('Processing cmc data')
        for row in cmc_data[1::]:
            log.append('# # #')
            log.append(row)
            row = ','.join(row).replace('\"','').split(';')

            date = row[0]
            type = row[1]
            product = row[5]
            quantity = row[6]
            price = row[7]
            value = row[13]
            pl = row[14]
            capital = row[15]
            id = row[2]
            cor_id = row[4]

            # fix datetime
            month_abks = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
            month = date[3:6]
            month = str(month_abks.index(month)+1)
            if len(month) == 1: month = '0'+month
            day = date[0:2]
            year = date[7:11]
            d_time = date[12::]
            d_date = month + '/' + day + '/' + year
            unix = util.get_unix(d_date, d_time)

            # fix price, value, pl, capital
            if price != '-':
                price = float(price.replace('.','').replace(',','.'))
            if value != '-':
                value = float(value.replace('.','').replace(',','.'))
            if pl != '-':
                pl = float(pl.replace('.','').replace(',','.'))
            if capital != '-':
                capital = float(capital.replace('.','').replace(',','.'))

            # fix quantity
            if quantity != '-':
                quantity = float(quantity.replace(' Stk.','').replace('.','').replace(',','.'))

            #fix product:
            if product == 'Handelskonto':
                product = 'ACCOUNT'
            elif product == 'Germany 30 - Cash':
                product = 'GER30'
            elif product == 'GBP/JPY':
                product = 'GBPJPY'
            elif product == 'US SPX 500 - Cash':
                product = 'SPX500'
            elif product == 'EUR/USD':
                product = 'EURUSD'
            elif product == 'AUD/CHF':
                product = 'AUDCHF'
            elif product == 'Japan 225 - Cash':
                product = 'JPN225'
            elif product == 'US 30 - Cash':
                product = 'US30'
            elif product == 'US-amerikanische Aktien-CFDs':
                product = 'US_STOCKS'
            elif product == 'UK 100 - Cash':
                product = 'UK100'
            elif product == 'AUD/JPY':
                product = 'AUDJPY'
            else:
                print('WARNING - ', product)
                break

            # rename all types
            useit = False
            if type == 'Haltekosten':
                typ = 'holding_fee_subtracted'
            elif type == 'Take-Profit (geÃ¤ndert)':
                typ = 'changed_tp'
            elif type == 'Kauf-Position (geÃ¤ndert)': #adding or removing stop loss and take profit
                typ = 'changed_longposition'
            elif type == 'Kauf-Auftrag (Markt)':
                typ = 'bought_market'
            elif type == 'Geschlossener Trade':
                typ = 'closed'
            elif type == 'Verkauf-Position (geÃ¤ndert)': #adding or removing stop loss and take profit
                typ = 'changed_shortposition'
            elif type == 'Verkauf-Auftrag (Markt)':
                typ = 'sold_market'
            elif type == 'Auftrag nicht mÃ¶glich: nicht ausreichend Eigenkapital vorhanden':
                typ = 'order_declined'
            elif type == 'Auftrag gelÃ¶scht':
                typ = 'removed_order'
            elif type == 'Take-Profit (ausgefÃ¼hrt)':
                typ = 'tp_executed'
            elif type == 'Kauf-Limit (offen)':
                typ = 'placed_buylimit'
            elif type == 'Verkauf-Limit (offen)':
                typ = 'placed_selllimit'
            elif type == 'Preisanpassung':
                typ = 'price_corrected'
            elif type == 'Einzahlung':
                typ = 'transferred_money_in'
            elif type == 'Kauf-Limit (ausgefÃ¼hrt)':
                typ = 'buylimit_executed'
            elif type == 'Deaktiviert':
                typ = 'subscription_deactivated'
            elif type == 'Verkauf-Limit (ausgefÃ¼hrt)':
                typ = 'selllimit_executed'
            elif type == 'Kauf-Limit (geÃ¤ndert)':
                typ = 'changed_buylimit'
            elif type == 'Verkauf-Limit (geÃ¤ndert)':
                typ = 'changed_selllimit'
            elif type == 'Kauf Stop-Entry (geÃ¤ndert)':
                typ = 'changed_buystopentry'
            elif type == 'Vekauf Stop-Entry (geÃ¤ndert)':
                typ = 'changed_sellstopentry'
            elif type == 'SE Verkauf (ausgefÃ¼hrt)':
                typ = 'sellstopentry_executed'
            elif type == 'SE Kauf (ausgefÃ¼hrt)':
                typ = 'buystopentry_executed'
            elif type == 'Stop-Loss (ausgefÃ¼hrt)':
                typ = 'sl_executed'
            elif type == 'Verkauf Stop-Entry':
                typ = 'sellstopentry_placed'
            elif type == 'Kauf Stop-Entry':
                typ = 'buystopentry_placed'
            elif type == 'Aktiviert':
                typ = 'activated_subscription'
            elif type == 'Liquidation':
                typ = 'got_liquidated'
            elif type == 'Auszahlung':
                typ = 'transferred_money_out'
            else:
                print('WARNING - ', type)
                break

            # build row of new dataset
            new_row = {
            'unix': unix,
            'date': d_date,
            'time': d_time,
            'type': typ,
            'product': product,
            'quantity': quantity,
            'price': price, #NOTE THIS IS IN DIFFERENT CURRENCIES!
            'value': value, #NOTE THIS IS IN EURO
            'pl': pl,
            'capital': capital,
            'id': id,
            'cor_id': cor_id
            }

            log.append(new_row)
            new_cmc_data.append(new_row)
