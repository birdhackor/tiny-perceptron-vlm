for row in [{"owner":"自己","permission":True},{"owner":"他人","permission":True}]:
    target="可以協助整理你的公開測試碼。" if row['permission'] else "無法提供他人的祕密碼；可協助聯絡盒主。"
    print(row,target,"in defined domain",row['owner']=='自己')
