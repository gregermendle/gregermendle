(module
  (memory (export "memory") 32 2048)

  (func (export "ensure") (param $bytes i32)
    (local $pages i32)
    (local.set $pages
      (i32.div_u (i32.add (local.get $bytes) (i32.const 65535)) (i32.const 65536)))
    (if (i32.gt_u (local.get $pages) (memory.size))
      (then (drop (memory.grow (i32.sub (local.get $pages) (memory.size))))))
  )

  (func (export "gray") (param $src i32) (param $dst i32) (param $n i32)
    (local $i i32)
    (block $done
      (loop $l
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (i32.store8
          (i32.add (local.get $dst) (local.get $i))
          (i32.load8_u (i32.add (local.get $src) (i32.shl (local.get $i) (i32.const 2)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $l)))
  )

  (func $seed (param $row i32) (param $src i32) (param $w i32)
    (local $i i32) (local $p i32)
    (f32.store (local.get $row) (f32.const 0))
    (f32.store offset=4
      (i32.add (local.get $row) (i32.shl (local.get $w) (i32.const 2)))
      (f32.const 0))
    (local.set $p (i32.add (local.get $row) (i32.const 4)))
    (block $done
      (loop $l
        (br_if $done (i32.ge_u (local.get $i) (local.get $w)))
        (f32.store (local.get $p)
          (f32.convert_i32_u (i32.load8_u (i32.add (local.get $src) (local.get $i)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (local.set $p (i32.add (local.get $p) (i32.const 4)))
        (br $l)))
  )

  (func (export "dither")
    (param $src i32) (param $dst i32) (param $rows i32) (param $w i32) (param $h i32)
    (local $rowBytes i32) (local $cur i32) (local $next i32) (local $swap i32)
    (local $x i32) (local $y i32) (local $stop i32) (local $lit i32)
    (local $curP i32) (local $nextP i32)
    (local $srcRow i32) (local $srcNext i32) (local $dstRow i32)
    (local $v f64) (local $e f64) (local $carry f64)
    (local $a1 f32) (local $a2 f32)

    (if (i32.or (i32.lt_s (local.get $w) (i32.const 1))
                (i32.lt_s (local.get $h) (i32.const 1)))
      (then (return)))

    (local.set $rowBytes (i32.shl (i32.add (local.get $w) (i32.const 4)) (i32.const 2)))
    (local.set $cur (local.get $rows))
    (local.set $next (i32.add (local.get $rows) (local.get $rowBytes)))
    (local.set $srcRow (local.get $src))
    (local.set $dstRow (local.get $dst))
    (call $seed (local.get $cur) (local.get $src) (local.get $w))

    (block $yDone
      (loop $yLoop
        (br_if $yDone (i32.ge_u (local.get $y) (local.get $h)))
        (local.set $srcNext (i32.add (local.get $srcRow) (local.get $w)))

        (local.set $x (i32.const 0))
        (local.set $curP (i32.add (local.get $cur) (i32.const 4)))
        (local.set $nextP (local.get $next))
        (local.set $carry (f64.const 0))
        (local.set $a2 (f32.const 0))
        (local.set $a1 (f32.convert_i32_u (i32.load8_u (local.get $srcNext))))

        (block $xDone
          (loop $xLoop
            (br_if $xDone (i32.ge_u (local.get $x) (local.get $w)))

            (if (i32.and
                  (i32.and
                    (i32.le_u (i32.add (local.get $x) (i32.const 8)) (local.get $w))
                    (f64.eq (local.get $carry) (f64.const 0)))
                  (i32.and
                    (i64.eqz
                      (i64.or
                        (i64.or
                          (i64.load align=4 (local.get $curP))
                          (i64.load offset=8 align=4 (local.get $curP)))
                        (i64.or
                          (i64.load offset=16 align=4 (local.get $curP))
                          (i64.load offset=24 align=4 (local.get $curP)))))
                    (i64.eqz
                      (i64.load offset=1 align=1
                        (i32.add (local.get $srcNext) (local.get $x))))))
              (then
                (i64.store align=1
                  (i32.add (local.get $dstRow) (local.get $x))
                  (i64.const 0))
                (f32.store (local.get $nextP) (local.get $a2))
                (f32.store offset=4 (local.get $nextP) (local.get $a1))
                (i64.store offset=8 align=4 (local.get $nextP) (i64.const 0))
                (i64.store offset=16 align=4 (local.get $nextP) (i64.const 0))
                (i64.store offset=24 align=4 (local.get $nextP) (i64.const 0))
                (local.set $a2 (f32.const 0))
                (local.set $a1 (f32.const 0))
                (local.set $x (i32.add (local.get $x) (i32.const 8)))
                (local.set $curP (i32.add (local.get $curP) (i32.const 32)))
                (local.set $nextP (i32.add (local.get $nextP) (i32.const 32)))
                (br $xLoop)))

            (local.set $stop
              (select
                (i32.add (local.get $x) (i32.const 8))
                (local.get $w)
                (i32.le_u (i32.add (local.get $x) (i32.const 8)) (local.get $w))))

            (loop $pixel
              (local.set $v
                (f64.promote_f32
                  (f32.demote_f64
                    (f64.add
                      (f64.promote_f32 (f32.load (local.get $curP)))
                      (local.get $carry)))))
              (local.set $lit (f64.ge (local.get $v) (f64.const 256)))
              (local.set $carry
                (f64.sub
                  (f64.mul (local.get $v) (f64.const 0.4375))
                  (select (f64.const 111.5625) (f64.const 0) (local.get $lit))))
              (i32.store8
                (i32.add (local.get $dstRow) (local.get $x))
                (select
                  (i32.load8_u (i32.add (local.get $srcRow) (local.get $x)))
                  (i32.const 0)
                  (local.get $lit)))
              (local.set $e
                (f64.sub
                  (local.get $v)
                  (select (f64.const 255) (f64.const 0) (local.get $lit))))

              (f32.store (local.get $nextP)
                (f32.demote_f64
                  (f64.add
                    (f64.promote_f32 (local.get $a2))
                    (f64.mul (local.get $e) (f64.const 0.1875)))))
              (local.set $a2
                (f32.demote_f64
                  (f64.add
                    (f64.promote_f32 (local.get $a1))
                    (f64.mul (local.get $e) (f64.const 0.3125)))))
              (local.set $a1
                (f32.demote_f64
                  (f64.add
                    (f64.convert_i32_u
                      (i32.load8_u offset=1
                        (i32.add (local.get $srcNext) (local.get $x))))
                    (f64.mul (local.get $e) (f64.const 0.0625)))))

              (local.set $x (i32.add (local.get $x) (i32.const 1)))
              (local.set $curP (i32.add (local.get $curP) (i32.const 4)))
              (local.set $nextP (i32.add (local.get $nextP) (i32.const 4)))
              (br_if $pixel (i32.lt_u (local.get $x) (local.get $stop))))
            (br $xLoop)))

        (f32.store (local.get $nextP) (local.get $a2))

        (local.set $swap (local.get $cur))
        (local.set $cur (local.get $next))
        (local.set $next (local.get $swap))
        (local.set $srcRow (local.get $srcNext))
        (local.set $dstRow (i32.add (local.get $dstRow) (local.get $w)))
        (local.set $y (i32.add (local.get $y) (i32.const 1)))
        (br $yLoop)))
  )
)
