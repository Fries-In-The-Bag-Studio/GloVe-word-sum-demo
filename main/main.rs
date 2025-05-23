use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

// Type alias for better readability: a word vector is just a Vec of f32
type WordVector = Vec<f32>;

fn main() {
    // Load word vectors from a GloVe .txt file
    let embeddings = load_glove_vectors("glove.6B.50d.txt");

    // Read the command-line arguments (words and operators like +, -)
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: cargo run -- word1 + word2 - word3 ...");
        return;
    }

    // Compute the resulting vector from the expression
    let result_vector = compute_expression_vector(&args, &embeddings);

    // Find the nearest word to the resulting vector
    if let Some((word, similarity)) = find_nearest_neighbor(&result_vector, &embeddings) {
        println!("Closest word: '{}' (cosine similarity: {:.4})", word, similarity);
    } else {
        println!("No nearest neighbor found.");
    }
}

/// Load the GloVe word vectors from the specified file into a HashMap.
fn load_glove_vectors(filename: &str) -> HashMap<String, WordVector> {
    let file = File::open(filename).expect("Could not open GloVe file");
    let reader = BufReader::new(file);
    let mut embeddings = HashMap::new();

    for line in reader.lines() {
        if let Ok(line) = line {
            let mut parts = line.split_whitespace();
            if let Some(word) = parts.next() {
                // Parse the rest of the line into f32 vector components
                let vector: WordVector = parts.filter_map(|s| s.parse::<f32>().ok()).collect();
                if vector.len() == 50 {
                    embeddings.insert(word.to_string(), vector);
                }
            }
        }
    }

    embeddings
}

/// Compute the resulting vector from an expression like "king + queen - man"
fn compute_expression_vector(args: &[String], embeddings: &HashMap<String, WordVector>) -> WordVector {
    // Initialize a zero vector with the same dimension as GloVe (50)
    let mut result = vec![0.0; 50];
    let mut current_op = 1.0; // 1.0 for addition, -1.0 for subtraction

    for token in args {
        match token.as_str() {
            "+" => current_op = 1.0,
            "-" => current_op = -1.0,
            word => {
                if let Some(vector) = embeddings.get(word) {
                    for i in 0..50 {
                        result[i] += current_op * vector[i]; // Add or subtract vector
                    }
                } else {
                    eprintln!("Warning: '{}' not in vocabulary, skipping.", word);
                }
            }
        }
    }

    result
}

/// Find the word whose vector has the highest cosine similarity with the given vector
fn find_nearest_neighbor(target: &WordVector, embeddings: &HashMap<String, WordVector>) -> Option<(String, f32)> {
    let mut best_word = None;
    let mut best_similarity = -1.0;

    for (word, vector) in embeddings {
        let sim = cosine_similarity(target, vector);
        if sim > best_similarity {
            best_similarity = sim;
            best_word = Some(word.clone());
        }
    }

    best_word.map(|w| (w, best_similarity))
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(vec1: &WordVector, vec2: &WordVector) -> f32 {
    let dot: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let norm1 = vec1.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm2 = vec2.iter().map(|v| v * v).sum::<f32>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot / (norm1 * norm2)
    }
}
