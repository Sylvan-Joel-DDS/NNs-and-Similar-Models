"""EcoSort waste-management pipeline.

Modules
-------
config      paths, constants, cache fingerprints
images      RealWaste discovery, stratified split, cached uint8 image arrays
cnn         MobileNetV2 transfer learning (cached frozen-feature head search, staged fine-tuning)
text_clf    description cleaning, splits, TF-IDF classifier (trained once, cached with joblib)
policy_rag  policy knowledge base, conflict detection, retriever, grounded answer composition
generator   flan-t5 wrapper (lazy PyTorch import)
rag_train   fine-tuning data + loop for the generator
assistant   EcoSortAssistant: image / text -> category -> instructions + policy documents
run_pipeline  `python -m Pipeline.run_pipeline --stage all` builds every cache from the command line
"""
