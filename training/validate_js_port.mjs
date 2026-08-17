import fs from "node:fs";

const MODELS_DIR = new URL("../docs/models/", import.meta.url);

function charWbNgrams(text, minN, maxN) {
  const ngrams = [];
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const w = " " + word + " ";
    const wLen = w.length;
    for (let n = minN; n <= maxN; n++) {
      let offset = 0;
      ngrams.push(w.slice(offset, offset + n));
      while (offset + n < wLen) {
        offset += 1;
        ngrams.push(w.slice(offset, offset + n));
      }
      if (offset === 0) break;
    }
  }
  return ngrams;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function tfidfPredict(text, model) {
  const normalized = text.toLowerCase();
  const ngrams = charWbNgrams(normalized, model.ngram_min, model.ngram_max);
  const counts = new Map();
  for (const g of ngrams) {
    const idx = model.vocab[g];
    if (idx === undefined) continue;
    counts.set(idx, (counts.get(idx) || 0) + 1);
  }
  let sumSquares = 0;
  const entries = [];
  for (const [idx, tf] of counts) {
    const val = tf * model.idf[idx];
    entries.push([idx, val]);
    sumSquares += val * val;
  }
  const norm = Math.sqrt(sumSquares) || 1;
  let z = model.intercept;
  for (const [idx, val] of entries) z += (val / norm) * model.coef[idx];
  return sigmoid(z);
}

function tokenize(text) {
  const matches = text.match(/[a-zA-ZäöüßÄÖÜ]+/g) || [];
  return matches.map((w) => w.toLowerCase());
}

function word2vecPredict(text, model) {
  const tokens = tokenize(text);
  const dim = model.dim;
  const acc = new Float64Array(dim);
  let count = 0;
  for (const t of tokens) {
    const idx = model.vocab[t];
    if (idx === undefined) continue;
    const base = idx * dim;
    for (let d = 0; d < dim; d++) acc[d] += model.vectors[base + d];
    count++;
  }
  if (count > 0) for (let d = 0; d < dim; d++) acc[d] /= count;
  let z = model.intercept;
  for (let d = 0; d < dim; d++) z += acc[d] * model.coef[d];
  return sigmoid(z);
}

async function main() {
  const tfidfVocabRaw = JSON.parse(fs.readFileSync(new URL("tfidf_vocab.json", MODELS_DIR)));
  const tfidfWeights = new Float32Array(fs.readFileSync(new URL("tfidf_weights.bin", MODELS_DIR)).buffer);
  const vocabSize = Object.keys(tfidfVocabRaw.vocab).length;
  const tfidfModel = {
    ngram_min: tfidfVocabRaw.ngram_min,
    ngram_max: tfidfVocabRaw.ngram_max,
    intercept: tfidfVocabRaw.intercept,
    vocab: tfidfVocabRaw.vocab,
    idf: tfidfWeights.subarray(0, vocabSize),
    coef: tfidfWeights.subarray(vocabSize, vocabSize * 2),
  };

  const w2vVocabRaw = JSON.parse(fs.readFileSync(new URL("word2vec_vocab.json", MODELS_DIR)));
  const w2vVectors = new Float32Array(fs.readFileSync(new URL("word2vec_vectors.bin", MODELS_DIR)).buffer);
  const w2vModel = {
    dim: w2vVocabRaw.dim,
    intercept: w2vVocabRaw.intercept,
    coef: w2vVocabRaw.coef,
    vocab: w2vVocabRaw.vocab,
    vectors: w2vVectors,
  };

  const testCases = JSON.parse(fs.readFileSync("/tmp/expected.json", "utf8"));

  let allMatch = true;
  for (const tc of testCases) {
    const tfidfP = tfidfPredict(tc.text, tfidfModel);
    const w2vP = word2vecPredict(tc.text, w2vModel);
    const tfidfMatch = Math.abs(tfidfP - tc.tfidf) < 1e-5;
    const w2vMatch = Math.abs(w2vP - tc.word2vec) < 1e-5;
    allMatch = allMatch && tfidfMatch && w2vMatch;
    console.log(tc.text.slice(0, 40));
    console.log(`  tfidf:    js=${tfidfP.toFixed(6)} py=${tc.tfidf.toFixed(6)} match=${tfidfMatch}`);
    console.log(`  word2vec: js=${w2vP.toFixed(6)} py=${tc.word2vec.toFixed(6)} match=${w2vMatch}`);
  }
  console.log(allMatch ? "\nALL MATCH" : "\nMISMATCH FOUND");
}

main();
