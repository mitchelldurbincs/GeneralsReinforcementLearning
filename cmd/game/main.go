package main

import (
    "fmt"
    "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
)

func main() {
    g := game.NewEngine(5, 5, 2)
    for turn := 0; turn < 10; turn++ {
        g.Step()
        //           %d  (int)           %s  (string)
        fmt.Printf("After turn %d (growth):\n%s\n\n",
            turn+1,            // matches %d
            g.Board(),         // matches %s
        )
    }
}