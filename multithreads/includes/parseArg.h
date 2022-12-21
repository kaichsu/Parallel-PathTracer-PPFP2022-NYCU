#include <stdio.h>
#include <algorithm>
#include <getopt.h>
#include <string.h>

void usage(const char *progname)
{
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -h  --height <INT>    image height\n");
    printf("  -w  --width <INT>     image width\n");
    printf("  -s  --samples <INT>   samples per pixel\n");
    printf("  -d  --maxDepth <INT>  the maximun times of a ray could bounce\n");
    printf("  -d  --threads <INT>   the amount of active threads\n");
    printf("  -v  --view 1/2        the view of the scene\n");
    printf("  -o  --output <STRING> image output path, default: ./image.png\n");
    printf("  -?  --help            This message\n");
    return;
}

int parse_arg(int argc, char **argv, int &height, int &width, int &samples, int &maxDepth, int &max_thread, int &view, const char **path)
{
    int opt;
    static struct option long_options[] = {
        {"height", 1, 0, 'h'},
        {"width", 1, 0, 'w'},
        {"samples", 1, 0, 's'},
        {"depth", 1, 0, 'd'},
        {"threads", 1, 0, 't'},
        {"view", 1, 0, 'v'},
        {"output", 1, 0, 'o'},
        {"help", 0, 0, '?'},
        {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "h:w:s:b:d:t:v:o:?", long_options, NULL)) != EOF)
    {
        switch (opt)
        {
        case 'h':
        {
            height = atoi(optarg);
            if (height <= 0){
                fprintf(stderr, "height should > 0\n");
                return 1;
            }
            break;
        }
        case 'w':
        {
            width = atoi(optarg);
            if (width <= 0){
                fprintf(stderr, "width should > 0\n");
                return 1;
            }
            break;
        }
        case 's':
        {
            samples = atoi(optarg);
            if (samples <= 0){
                fprintf(stderr, "samples should > 0\n");
                return 1;
            }
            break;
        }
        case 'd':
        {
            maxDepth = atoi(optarg);
            if (maxDepth <= 0){
                fprintf(stderr, "maxDepth should > 0\n");
                return 1;
            }
            break;
        }
        case 't':
        {
            max_thread = atoi(optarg);
            if (max_thread <= 0){
                fprintf(stderr, "max_thread should > 0\n");
                return 1;
            }
            break;
        }
        case 'o':
        {
            *path = optarg;
            if (strlen(*path) <= 0){
                fprintf(stderr, "output path shouldn't be empty\n");
                return 1;
            }
            break;
        }
        case 'v':
        {
            view = atoi(optarg);
            if (!(view == 1 || view == 2)){
                fprintf(stderr, "view must 1 or 2\n");
                return 1;
            }
            break;
        }
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    return 0;
}