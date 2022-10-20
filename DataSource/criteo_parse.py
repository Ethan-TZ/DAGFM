from tqdm import tqdm
import math

def trans_int_feat(val):
  return int(math.ceil(math.log(val)**2+1))

def pre_parse_line(line, feat_list, int_feat_cnt=13, cate_feat_cnt=26, with_int_feat=False):
  fields_cnt = int_feat_cnt + cate_feat_cnt
  splits = line.rstrip('\n').split('\t', fields_cnt+1)

  start_index = 0 if with_int_feat else int_feat_cnt
  for idx in range(start_index, fields_cnt):
    val = splits[idx+1]
    if val == '':
      continue
    elif idx < int_feat_cnt:
      val = int(val)
      if val > 2:
        val = trans_int_feat(val)
    else:
        val = int(val, 16)

    if val not in feat_list[idx]:
      feat_list[idx][val] = 1
    else:
      feat_list[idx][val] += 1

  return

def parse_line(line, feat_list, int_feat_cnt=13, cate_feat_cnt=26, with_int_feat = False):
  fields_cnt = int_feat_cnt + cate_feat_cnt
  splits = line.rstrip('\n').split('\t', fields_cnt+1)

  label = int(splits[0])
  vals = []

  start_index = 0 if with_int_feat else int_feat_cnt
  for idx in range(start_index, fields_cnt):
    val = splits[idx+1]
    if val == '':
      vals.append(0)
      continue
    elif idx < int_feat_cnt:
      val = int(val)
      if val > 2:
        val = trans_int_feat(val)
    else:
      val = int(val, 16)

    if val not in feat_list[idx]:
      vals.append(0)
    else:
      vals.append(feat_list[idx][val])
  return label, vals

if __name__ == "__main__":
  thres = 8
  int_feat_cnt = 13
  cate_feat_cnt = 26
  with_int_feat = True

  data_file = 'Criteo.csv'
  out_file = open('Criteo_all.csv', 'w')

  dataset = open(data_file, 'r').readlines()

  feat_list = []
  for i in range(40):
    feat_list.append({})

  for line in tqdm(dataset):
    pre_parse_line(line, feat_list, int_feat_cnt, cate_feat_cnt, with_int_feat)

  for lst in tqdm(feat_list[:int_feat_cnt]):
    idx = 1
    for key, val in lst.items():
      lst[key] = idx
      idx += 1

  for lst in tqdm(feat_list[int_feat_cnt:]):
    idx = 1
    for key, val in lst.items():
      if val < thres:
        del lst[key]
      else:
        lst[key] = idx
        idx += 1


  for line in tqdm(dataset):
    key, vals = parse_line(line, feat_list, int_feat_cnt, cate_feat_cnt, with_int_feat)
    if vals is None:
      continue
    out_file.write('%s,%s\n' % (key, ','.join([str(s) for s in vals])))


  out_file.close()
  del dataset
