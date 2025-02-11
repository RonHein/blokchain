import os

def split_jsonl(input_path, chunk_size=150 * 1024 * 1024, output_prefix='chunk_'):
    with open(input_path, 'r', encoding='utf-8') as infile:
        chunk_index = 1
        current_chunk_size = 0
        outfile_path = f"{output_prefix}{chunk_index}.jsonl"
        outfile = open(outfile_path, 'w', encoding='utf-8')

        for line in infile:
            line_size = len(line.encode('utf-8'))

            # If this line would exceed limit, start a new file
            if current_chunk_size + line_size > chunk_size:
                outfile.close()
                chunk_index += 1
                current_chunk_size = 0
                outfile_path = f"{output_prefix}{chunk_index}.jsonl"
                outfile = open(outfile_path, 'w', encoding='utf-8')

            outfile.write(line)
            current_chunk_size += line_size

        outfile.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python split_jsonl.py <input.jsonl> [chunk_size_in_MB] [output_prefix]")
        sys.exit(1)

    input_file = sys.argv[1]
    if len(sys.argv) > 2:
        mb_size = int(sys.argv[2])
    else:
        mb_size = 150  # default to 150 MB

    if len(sys.argv) > 3:
        prefix = sys.argv[3]
    else:
        prefix = "chunk_"

    split_jsonl(input_file, mb_size * 1024 * 1024, prefix)
