(module
  (memory (export "memory") 32 2048)
  (func (export "ensure") (param $bytes i32)
    (local $pages i32)
    (local.set $pages
      (i32.div_u (i32.add (local.get $bytes) (i32.const 65535)) (i32.const 65536)))
    (if (i32.gt_u (local.get $pages) (memory.size))
      (then (drop (memory.grow (i32.sub (local.get $pages) (memory.size))))))
  )
  (func (export "dither")
    (param $src i32) (param $work i32) (param $dst i32)
    (param $w i32) (param $h i32)
    (local $n i32) (local $i i32) (local $x i32) (local $y i32)
    (local $srcPtr i32) (local $workPtr i32)
    (local $lastX i32) (local $lastY i32)
    (local $oldVal f64) (local $err f64)
    (local $idx4 i32) (local $sum i32)

    (local.set $n (i32.mul (local.get $w) (local.get $h)))
    (local.set $lastX (i32.sub (local.get $w) (i32.const 1)))
    (local.set $lastY (i32.sub (local.get $h) (i32.const 1)))
    (local.set $srcPtr (local.get $src))
    (local.set $workPtr (local.get $work))

    (block $lumDone
      (loop $lum
        (if (i32.ge_u (local.get $i) (local.get $n)) (then (br $lumDone)))
        (local.set $sum
          (i32.add
            (i32.add
              (i32.load8_u (local.get $srcPtr))
              (i32.load8_u offset=1 (local.get $srcPtr)))
            (i32.load8_u offset=2 (local.get $srcPtr))))
        (f32.store (local.get $workPtr)
          (f32.demote_f64
            (f64.mul (f64.convert_i32_u (local.get $sum)) (f64.const 0.3333333333333333))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (local.set $srcPtr (i32.add (local.get $srcPtr) (i32.const 4)))
        (local.set $workPtr (i32.add (local.get $workPtr) (i32.const 4)))
        (br $lum)
      )
    )

    (local.set $i (i32.const 0))
    (local.set $y (i32.const 0))
    (block $yDone
      (loop $yLoop
        (if (i32.ge_u (local.get $y) (local.get $h)) (then (br $yDone)))
        (local.set $x (i32.const 0))
        (block $xDone
          (loop $xLoop
            (if (i32.ge_u (local.get $x) (local.get $w)) (then (br $xDone)))
            (local.set $idx4 (i32.shl (local.get $i) (i32.const 2)))
            (local.set $oldVal
              (f64.promote_f32 (f32.load (i32.add (local.get $work) (local.get $idx4)))))
            (if (f64.lt (local.get $oldVal) (f64.const 256))
              (then
                (i32.store (i32.add (local.get $dst) (local.get $idx4)) (i32.const 0xff000000))
                (if (f64.ne (local.get $oldVal) (f64.const 0))
                  (then
                    (call $diffuse
                      (local.get $work) (local.get $i) (local.get $x) (local.get $y)
                      (local.get $w) (local.get $lastX) (local.get $lastY) (local.get $oldVal))
                  )
                )
              )
              (else
                (local.set $err (f64.sub (local.get $oldVal) (f64.const 255)))
                (i32.store (i32.add (local.get $dst) (local.get $idx4))
                  (i32.or
                    (i32.load (i32.add (local.get $src) (local.get $idx4)))
                    (i32.const 0xff000000)))
                (call $diffuse
                  (local.get $work) (local.get $i) (local.get $x) (local.get $y)
                  (local.get $w) (local.get $lastX) (local.get $lastY) (local.get $err))
              )
            )
            (local.set $x (i32.add (local.get $x) (i32.const 1)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $xLoop)
          )
        )
        (local.set $y (i32.add (local.get $y) (i32.const 1)))
        (br $yLoop)
      )
    )
  )
  (func $diffuse
    (param $work i32) (param $i i32) (param $x i32) (param $y i32)
    (param $w i32) (param $lastX i32) (param $lastY i32) (param $err f64)
    (if (i32.lt_u (local.get $x) (local.get $lastX))
      (then (call $accum (local.get $work) (i32.add (local.get $i) (i32.const 1)) (local.get $err) (f64.const 0.4375))))
    (if (i32.lt_u (local.get $y) (local.get $lastY))
      (then
        (if (i32.gt_u (local.get $x) (i32.const 0))
          (then (call $accum (local.get $work) (i32.sub (i32.add (local.get $i) (local.get $w)) (i32.const 1)) (local.get $err) (f64.const 0.1875))))
        (call $accum (local.get $work) (i32.add (local.get $i) (local.get $w)) (local.get $err) (f64.const 0.3125))
        (if (i32.lt_u (local.get $x) (local.get $lastX))
          (then (call $accum (local.get $work) (i32.add (i32.add (local.get $i) (local.get $w)) (i32.const 1)) (local.get $err) (f64.const 0.0625))))
      )
    )
  )
  (func $accum (param $work i32) (param $i i32) (param $err f64) (param $weight f64)
    (local $ptr i32)
    (local.set $ptr (i32.add (local.get $work) (i32.shl (local.get $i) (i32.const 2))))
    (f32.store (local.get $ptr)
      (f32.demote_f64
        (f64.add
          (f64.promote_f32 (f32.load (local.get $ptr)))
          (f64.mul (local.get $err) (local.get $weight)))))
  )
)
