import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--statuses', nargs='+', help="The outcome of each of the tests [success, failure, cancelled, or skipped]", required=True)
    parser.add_argument('--reasons', nargs='+', help="The reasons to print in case of failure", required=True)
    p_args = parser.parse_args()

    assert len(p_args.statuses) == len(p_args.reasons)

    summary = "Congratulations! Tests are passing."
    for status, reason in zip(p_args.statuses, p_args.reasons):
        if status == "failure":
            summary = reason
            break
        elif status != "success":
            summary = "Tests cancelled"
            break

    output_file = 'test_json_result.json'

    json_ouput = {
        "summary":summary
    }

    with open(output_file, 'w') as f:
        json.dump(json_ouput,f)
