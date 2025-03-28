from natasha import (
    Segmenter,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    
    Doc
)
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

text = 'Посол Израиля на Украине Йоэль Лион признался, что пришел в шок, узнав о решении властей Львовской области объявить 2019 год годом лидера запрещенной в России Организации украинских националистов (ОУН) Степана Бандеры...'
doc = Doc(text)

doc.segment(segmenter)
doc.tag_morph(morph_tagger)
doc.parse_syntax(syntax_parser)


from natasha import MorphVocab

morph_vocab = MorphVocab()

for token in doc.tokens:
    token.normalize(morph_vocab)

print([_.lemma for _ in doc.tokens])

#from nltk.corpus import stopwords

#words_pack = stopwords.words("russian")

#print([word.lemma for word in doc.tokens if word.lemma not in words_pack])