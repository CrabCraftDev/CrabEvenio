use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{parse2, Data, DeriveInput, Error, Result};

pub(crate) fn derive_component_set(input: TokenStream) -> Result<TokenStream> {
    let input = parse2::<DeriveInput>(input)?;
    let name = &input.ident;

    let struct_ = match input.data {
        Data::Struct(struct_) => struct_,
        Data::Enum(_) => {
            return Err(Error::new(
                Span::call_site(),
                "cannot derive `ComponentSet` on enums",
            ))
        }
        Data::Union(_) => {
            return Err(Error::new(
                Span::call_site(),
                "cannot derive `ComponentSet` on unions",
            ))
        }
    };

    let field_names = struct_.fields.iter().enumerate().map(|(idx, f)| {
        f.ident
            .clone()
            .map(|i| quote! { #i })
            .unwrap_or(quote! { #idx })
    });
    let field_types = struct_.fields.iter().map(|f| &f.ty).collect::<Vec<_>>();

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        #[automatically_derived]
        unsafe impl #impl_generics ::evenio::component::ComponentSet for #name #ty_generics #where_clause {
            #[inline]
            fn len() -> usize {
                0 #(+ #field_types::len())*
                // 0
            }

            fn add_components(world: &mut World, mut add_component_id: impl FnMut(::evenio::component::ComponentId)) {
                #(
                    #field_types::add_components(world, &mut add_component_id);
                )*
            }

            fn remove_components(world: &mut World, mut add_component_info: impl FnMut(::evenio::component::ComponentInfo)) {
                #(
                    #field_types::remove_components(world, &mut add_component_info);
                )*
            }

            fn get_components(&self, out: &mut ::evenio::component::ComponentPointerConsumer) {
                #(
                    self.#field_names.get_components(out);
                )*
            }
        }
    })
}
