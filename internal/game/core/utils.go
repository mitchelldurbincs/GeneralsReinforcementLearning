package core

import "fmt"

// IntToStringFixedWidth converts an integer to a string of a specified width,
// left-padding with spaces if the number string is shorter than the width.
// If the number string is longer than the width, it will not be truncated by this function's padding,
// but fmt.Sprintf's behavior for %*d will still try to print the full number.
// For display purposes where overflow is an issue, higher-level logic (like in Board()) should handle it.
func IntToStringFixedWidth(num int, width int) string {
	return fmt.Sprintf("%*d", width, num)
}

func GetActionType(action Action) string {
    if action == nil {
        return "nil"
    }
    return fmt.Sprintf("%T", action)
}
