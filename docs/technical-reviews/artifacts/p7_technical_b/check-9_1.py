record={'提問':'盒中有幾顆球？','回答':'目前沒有數量資訊，請提供球數。','有用':True,'誠實':True,'安全':True}
pair={'回答0安全':False,'回答1安全':False,'相對較安全':0}
print(record['有用'],record['誠實'],record['安全']);print('較安全回答可直接當安全正例',pair[f"回答{pair['相對較安全']}安全"])
assert all(record[x] for x in ['有用','誠實','安全']) and not pair[f"回答{pair['相對較安全']}安全"]
pair['相對較安全']=1;assert not pair[f"回答{pair['相對較安全']}安全"]
print('交換相對較安全索引仍非安全',pair[f"回答{pair['相對較安全']}安全"])
