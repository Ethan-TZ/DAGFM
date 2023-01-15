import time
featmap = {}
featidx = {}
user_feat = {}
item_feat = {}
field2type = {}
field2seqlen = {}

def trans_date_feat(datetime_str):
    datetime_str = time.localtime(float(datetime_str))
    return datetime_str.tm_wday, datetime_str.tm_hour

def parse_file(filename):
    col = []
    with open(filename,'r') as f:
        text = f.readlines().__iter__()
        title = next(text).strip('\n')
        for field in title.split('\t'):
            fname , ftype = field.split(':')
            col.append(fname)
            if fname not in field2type and 'token' in ftype:
                field2type[fname] = ftype
                featmap[fname] = {}
                featidx[fname] = 1
                if 'seq' in ftype:
                    field2seqlen[fname + '_len'] = 0
        for raw_line in text:
            raw_line = raw_line.strip('\n')
            pline = []
            for idx , data in enumerate(raw_line.split('\t')):
                fname = col[idx]
                mps = featmap[fname]
                if field2type[fname] == 'token':
                    if data not in mps:
                        mps[data] = str(featidx[fname])
                        featidx[fname] += 1
                    pline.append(mps[data])
                elif field2type[fname] == 'token_seq':
                    seq = []
                    for token in data.split(' '):
                        if token not in mps:
                            mps[token] = str(featidx[fname])
                            featidx[fname] += 1
                        seq.append(mps[token])
                    field2seqlen[fname + '_len'] = max(field2seqlen[fname + '_len'] , len(seq))
                    pline.append(' '.join(seq))
                elif field2type[fname] == 'float':
                    pline.append(data)
            for i in range(len(pline)):
                if type(pline[i]) != 'str':
                    pline[i] = str(pline[i])
            if 'user' in filename:
                user_feat[pline[0]] = pline[1:]
            else:
                item_feat[pline[0]] = pline[1:]
    return col

col_user = parse_file('./ml-1m.user')
col_item = parse_file('./ml-1m.item')

col_user.remove('user_id')
col_item.remove('item_id')

output = open('./movie_all.csv','w')
with open('./ml-1m.inter','r') as f:
    text = f.readlines().__iter__()
    title = next(text)
    col = []
    for field in title.strip('\n').split('\t'):
        fname , ftype = field.split(':')
        if fname == 'rating':
            fname = 'label'
        if fname == 'timestamp':
            continue
        col.append(fname)
    col.extend(['weekday','hour'])
    col.extend(col_user)
    col.extend(col_item)
    #output.write(','.join(col) + '\n')
    for raw_line in text:
        valid = True
        raw_line = raw_line.strip('\n')
        pline = []
        for idx , data in enumerate(raw_line.split('\t')):
            if idx == 0:
                pline.append(featmap['user_id'][data])
            elif idx == 1:
                pline.append(featmap['item_id'][data])
            elif idx == 2:
                if float(data) == 3.:
                    valid = False
                pline.append('1' if float(data) >= 4 else '0')
            elif idx == 3:
                weekday , hour = trans_date_feat(data)
                pline.extend([weekday , hour])
        for i in range(len(pline)):
            if type( pline[i] ) != str:
                pline[i] = str( pline[i] )
        pline.extend(user_feat[ pline[0] ])
        pline.extend(item_feat[ pline[1] ])
        if valid:
            output.write(','.join(pline) + '\n')
output.close()
