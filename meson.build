project(
  'pyccel',
  'c', 'fortran',
  version: '2.0.1',
  license: 'MIT',
  meson_version: '>= 1.5.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'python.install_env=auto'
  ],
)

python = import('python').find_installation(pure: false)

stc_src = meson.current_source_dir() / 'pyccel' / 'extensions' / 'STC'
gftl_src = meson.current_source_dir() / 'pyccel' / 'extensions' / 'gFTL'
buildroot = meson.project_build_root()
pkgroot = python.get_install_dir()

py_sources = run_command(
  python.full_path(), 'install_scripts/collect_files_to_install.py',
  check: true
).stdout().strip().split('\n')

python.install_sources(py_sources, preserve_path: true)


ff = meson.get_compiler('fortran')
cmake_cmd = find_program('cmake')

if cmake_cmd.found()
  build_dir = buildroot / 'gftl-build'
  install_dir = buildroot / 'gFTL'
  dest_dir = pkgroot / 'pyccel' / 'extensions'

  # 1) CMake setup
  setup = run_command(
      cmake_cmd,
      '-S', gftl_src,
      '-B', build_dir,
      '-DCMAKE_BUILD_TYPE=Release',
      '-DCMAKE_INSTALL_PREFIX=' + install_dir,
    check: true,
  )

  # 2) CMake compile
  compile_target = custom_target(
    'gftl-compile',
    command: [cmake_cmd, '--build', build_dir],
    output: 'compile-build'
  )

  # 3) CMake install (into dest_dir)
  install_target = custom_target(
    'gftl',
    command: [cmake_cmd, '--install', build_dir],
    input: compile_target,
    output: 'gFTL',
    install: true,
    install_dir: dest_dir
  )
else
  warning('Couldn\'t find CMake. Containers (list/set/dict) will not be available for Fortran.')
endif




# List of compilers to try
toolchains = [
  'gcc',
  'icx',
  'clang',
  'nvc',
]

meson_cmd = find_program('meson')
ninja_cmd = find_program('ninja', required : false)  # meson will choose backend

foreach tc : toolchains
  probe = find_program(tc, required : false)

  if not probe.found()
    message('Skipping ' + tc + ' (compiler not found)')
    continue
  endif

  message('Building STC with ' + tc)

  # Build directory for this toolchain
  build_dir = buildroot / ('stc-build-' + tc)
  install_dir = buildroot / ('stc-install-' + tc)
  dest_dir = pkgroot / 'pyccel' / 'extensions'

  # 1) meson setup
  setup = run_command(
      meson_cmd,
      'setup', build_dir, stc_src,
      '--buildtype=release',
      '--prefix=' + install_dir,
      '--backend=ninja',
    check: true,
    env: {'CC': tc}
  )

  # 2) meson compile
  compile_target = custom_target(
    'stc-compile-' + tc,
    command: [
      meson_cmd,
      'compile', '-C', build_dir
    ],
    output: 'stc-compile-build-' + tc
  )

  # 3) meson install (into dest_dir)
  install_target = custom_target(
    'stc-' + tc,
    command: [
      meson_cmd,
      'install', '-C', build_dir
    ],
    input: compile_target,
    output: 'stc-install-' + tc,
    install: true,
    install_dir: dest_dir
  )
endforeach

