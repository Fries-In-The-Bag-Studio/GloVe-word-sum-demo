use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::f32;

// Define a type alias for better readability
type WordVec = Vec<f32>;

/// Loads a GloVe-style vector file into a HashMap
fn load_glove_vectors(path: &str) -> HashMap<String, WordVec> {
    let file = File::open(path).expect("Unable to open file");
    let reader = BufReader::new(file);
    let mut vectors = HashMap::new();

    // Read each line in the file
    for line in reader.lines() {
        if let Ok(l) = line {
            // Split the line into word and its 50 floats
            let mut parts = l.split_whitespace();
            if let Some(word) = parts.next() {
                // Parse all the floats
                let vec: WordVec = parts.map(|x| x.parse::<f32>().unwrap()).collect();
                vectors.insert(word.to_string(), vec);
            }
        }
    }

    vectors
}

/// Computes cosine similarity between two vectors
fn cosine_similarity(a: &WordVec, b: &WordVec) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-10)  // add epsilon to avoid division by zero
}

/// Adds multiple vectors together element-wise
fn sum_vectors(vectors: Vec<&WordVec>) -> WordVec {
    let mut sum = vec![0.0; vectors[0].len()];
    for vec in vectors {
        for (i, val) in vec.iter().enumerate() {
            sum[i] += val;
        }
    }
    sum
}

/// Finds the nearest neighbor word to a given vector,
/// excluding the input words themselves
fn find_nearest_neighbor<'a>(
    sum_vec: &WordVec,
    vectors: &'a HashMap<String, WordVec>,
    exclude_words: &[String],
) -> Option<(&'a String, f32)> {
    let mut best_word = None;
    let mut best_score = -f32::INFINITY;

    for (word, vec) in vectors.iter() {
        // Skip input words
        if exclude_words.contains(word) {
            continue;
        }

        let similarity = cosine_similarity(sum_vec, vec);
        if similarity > best_score {
            best_score = similarity;
            best_word = Some(word);
        }
    }

    best_word.map(|w| (w, best_score))
}

fn main() {
    // Usage: cargo run glove.txt word1 word2 word3 ...
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <glove.txt> word1 word2 ...", args[0]);
        return;
    }

    let glove_path = &args[1];
    let input_words: Vec<String> = args[2..].to_vec();

    println!("Loading GloVe vectors...");
    let glove = load_glove_vectors(glove_path);

    // Collect vectors for all valid input words
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

    // Sum the vectors of the valid words
    let sum_vec = sum_vectors(found_vectors);

    // Find the nearest neighbor that isn't one of the input words
    if let Some((nearest_word, similarity)) = find_nearest_neighbor(&sum_vec, &glove, &input_words) {
        println!("Nearest neighbor: {} (similarity: {:.4})", nearest_word, similarity);
    } else {
        println!("No nearest neighbor found.");
    }
}
