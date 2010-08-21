#ifndef VOCABULARY_TREE_DATABASE_H
#define VOCABULARY_TREE_DATABASE_H

/// @todo Include some basic_types.h instead
#include <vocabulary_tree/vocabulary_tree.h>
#include <map>

namespace vt {

typedef uint32_t DocId;
typedef std::vector<Word> Document;

// score is in [0,2], where 0 is best and 2 is worst.
struct Match
{
  DocId id;
  float score;

  Match() {}
  Match(DocId _id, float _score) : id(_id), score(_score) {}

  bool operator<(const Match& other) const
  {
    return score < other.score;
  }
};
typedef std::vector<Match> Matches;

class Database
{
public:
  Database(uint32_t num_words = 0);

  // Insert a new document. The returned DocId identifies it.
  DocId insert(const Document& words);

  // Find the top N matches in the database for the query document.
  void find(const Document& words, size_t N, Matches& matches) const;

  // Find the top N matches, then insert the query document.
  DocId findAndInsert(const Document& words, size_t N, Matches& matches);

  // Compute the TF-IDF weights of all the words. To be called after
  // inserting a corpus of training examples into the database.
  void computeTfIdfWeights(float default_weight = 1.0f);

  void saveWeights(const std::string& file) const;
  void loadWeights(const std::string& file);

  // Save weights and documents
  //void save(const std::string& file) const;
  //void load(const std::string& file);

private:
  struct WordFrequency
  {
    DocId id;
    uint32_t count;

    WordFrequency(DocId _id, uint32_t _count) : id(_id), count(_count) {}
  };
  
  // Stored in increasing order by DocId
  typedef std::vector<WordFrequency> InvertedFile;

  /// @todo Use sorted vector?
  // typedef std::vector< std::pair<Word, float> > DocumentVector;
  typedef std::map<Word, float> DocumentVector;

  std::vector<InvertedFile> word_files_;
  std::vector<float> word_weights_;
  std::vector<DocumentVector> database_vectors_; // Precomputed for inserted documents

  void computeVector(const Document& words, DocumentVector& v) const;
  
  static void normalize(DocumentVector& v);
  static float sparseDistance(const DocumentVector& v1, const DocumentVector& v2);
};

} //namespace vt

#endif
