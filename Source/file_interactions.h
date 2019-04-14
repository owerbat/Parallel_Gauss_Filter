#include <fstream>


bool is_file_exist(const std::string &filename) {
    std::ifstream infile(filename.c_str());
    return infile.good();
}

std::string get_short_name(const std::string &filename) {
    std::string slash = filename.rfind("\\") != std::string::npos ? "\\" : "/";
	size_t position = filename.rfind(slash) + 1;
	std::string result = "";

	for (unsigned i = position; i < filename.length(); ++i)
		result += filename[i];

	return result;
}
