# Encode for cat_cols
import pickle
def transformer(list_):
  encode = LabelEncoder() 
  new_list = encode.fit_transform(list_)
  return new_list

#   for i in df.columns:
#     if df[i].dtype == 'object':
#       df[i] = encode.fit_transform(df[i])
#       return df