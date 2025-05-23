use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::f32;

/// Define a type alias for a word vector for clarity
type WordVec = Vec<f32>;

/// Loads GloVe vectors from a file into a HashMap
fn load_glove_vectors(path: &str) -> HashMap<String, WordVec> {
    let file = File::open(path).expect("Unable to open file");
    let reader = BufReader::new(file);
    let mut vectors = HashMap::new();

    for line in reader.lines() {
        if let Ok(l) = line {
            let mut parts = l.split_whitespace();
            if let Some(word) = parts.next() {
                let vec: WordVec = parts.map(|x| x.parse::<f32>().unwrap()).collect();
                vectors.insert(word.to_string(), vec);
            }
        }
    }

    vectors
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &WordVec, b: &WordVec) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-10) // epsilon to avoid divide-by-zero
}

/// Compute Euclidean distance between two vectors
fn euclidean_distance(a: &WordVec, b: &WordVec) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Averages a list of word vectors element-wise
fn average_vectors(vectors: Vec<&WordVec>) -> WordVec {
    let mut sum = vec![0.0; vectors[0].len()];
    let count = vectors.len() as f32;

    for vec in vectors {
        for (i, val) in vec.iter().enumerate() {
            sum[i] += val;
        }
    }

    for val in &mut sum {
        *val /= count;
    }

    sum
}

/// Finds the most similar word using cosine similarity or Euclidean distance
fn find_nearest_neighbor<'a>(
    target_vec: &WordVec,
    vectors: &'a HashMap<String, WordVec>,
    exclude_words: &[String],
    use_cosine: bool,
) -> Option<(&'a String, f32)> {
    let mut best_word = None;
    let mut best_score = if use_cosine { -f32::INFINITY } else { f32::INFINITY };

    for (word, vec) in vectors.iter() {
        if exclude_words.contains(word) {
            continue;
        }

        let score = if use_cosine {
            cosine_similarity(target_vec, vec)
        } else {
            euclidean_distance(target_vec, vec)
        };

        let is_better = if use_cosine {
            score > best_score
        } else {
            score < best_score
        };

        if is_better {
            best_score = score;
            best_word = Some(word);
        }
    }

    best_word.map(|w| (w, best_score))
}

fn main() {
    // Example: cargo run glove.txt word1 word2 --cosine or --euclidean
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: {} <glove.txt> word1 word2 ... [--cosine | --euclidean]", args[0]);
        return;
    }

    let glove_path = &args[1];

    // Identify if user selected cosine or Euclidean comparison
    let mode_arg = args.last().unwrap();
    let use_cosine = match mode_arg.as_str() {
        "--cosine" => true,
        "--euclidean" => false,
        _ => {
            eprintln!("Final argument must be either --cosine or --euclidean");
            return;
        }
    };

    let input_words: Vec<String> = args[2..args.len() - 1].to_vec();

    println!("Loading GloVe vectors...");
    let glove = load_glove_vectors(glove_path);

    // Gather all vectors for the given input words
    let mut found_vectors = Vec::new();
    for word in &input_words {
        if let Some(vec) = glove.get(word) {
            found_vectors.push(vec);
        } else {
            println!("Skipping unknown word: {}", word);
        }
    }

    if found_vectors.is_empty() {
        println!("No valid input words found in the database.");
        return;
    }

    // Compute the average vector of the input words
    let avg_vec = average_vectors(found_vectors);

    // Find the most similar word (not including the input words)
    if let Some((nearest_word, score)) =
        find_nearest_neighbor(&avg_vec, &glove, &input_words, use_cosine)
    {
        if use_cosine {
            println!("Most similar word (cosine): {} (similarity: {:.4})", nearest_word, score);
        } else {
            println!("Most similar word (euclidean): {} (distance: {:.4})", nearest_word, score);
        }
    } else {
        println!("No nearest neighbor found.");
    }
}
